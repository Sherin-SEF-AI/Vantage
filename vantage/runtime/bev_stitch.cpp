// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sherin Joseph Roy. Licensed under the Apache License 2.0 (LICENSE in this folder), unlike the
// rest of Vantage (GPL-3.0-or-later): embed it in proprietary and production code freely.
//
// Vantage reference bird's-eye stitcher. C++17, no dependencies; OpenMP when available.
//
// Loads the remap tables Vantage exports (bev_lut.txt, <tag>_map.bin, <tag>_weight.bin, body_mask.bin) and stitches
// one frame per camera into the bird's-eye view, optionally balancing the cameras' exposure from the ground they share
// first. It is the reference to check a production pipeline (NPP, VPI, CUDA, GL) against: its output matches
// Vantage's own stitcher to within a grey level. The vbev namespace is the part to embed in a capture loop.
//
//   g++ -O3 -std=c++17 -fopenmp bev_stitch.cpp -o bev_stitch          (or: cmake -B build && cmake --build build)
//   ./bev_stitch ..  out.ppm  F.ppm R.ppm ML.ppm MR.ppm QL.ppm QR.ppm   frames in manifest order, binary PGM/PPM
//   ./bev_stitch ..  out.ppm  --frames captures/ --balance --bench 200

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbev {

struct Image {
    int w = 0, h = 0, c = 0;              // c = 1 (grey) or 3 (RGB)
    std::vector<uint8_t> px;              // row-major, channels interleaved
};

struct Camera {
    std::string tag;
    int sensor_w = 0, sensor_h = 0;
    std::vector<float> map;               // width * height * 2: x, y in sensor pixels (pixel centre at i + 0.5), -1 = not seen
    std::vector<uint8_t> weight;          // width * height, 0..255; all cameras sum to ~255 where the ground is covered
};

struct Lut {
    int width = 0, height = 0;
    double resolution = 0;                // metres per output pixel
    std::vector<Camera> cams;
    std::vector<uint8_t> body;            // width * height, 1 under the vehicle
};

using Gains = std::vector<std::array<float, 3>>;

const uint8_t BODY_RGB[3] = {38, 38, 36};
const uint8_t EMPTY_RGB[3] = {20, 20, 20};

static std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T>
static std::vector<T> read_array(const std::string& path, size_t n) {
    std::vector<uint8_t> b = read_all(path);
    if (b.size() != n * sizeof(T))
        throw std::runtime_error(path + ": " + std::to_string(b.size()) + " bytes, expected " + std::to_string(n * sizeof(T)));
    std::vector<T> v(n);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

// bev_lut.txt: "width W", "height H", "resolution R", "camera TAG MAP WEIGHT SENSOR_W SENSOR_H" ..., "body FILE"
inline Lut load(const std::string& dir) {
    Lut L;
    std::ifstream f(dir + "/bev_lut.txt");
    if (!f) throw std::runtime_error("cannot open " + dir + "/bev_lut.txt");
    struct Entry { std::string tag, map, weight; int sw = 0, sh = 0; };
    std::vector<Entry> entries;
    std::string line, body;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        std::string key;
        s >> key;
        if (key == "width") s >> L.width;
        else if (key == "height") s >> L.height;
        else if (key == "resolution") s >> L.resolution;
        else if (key == "camera") { Entry e; s >> e.tag >> e.map >> e.weight >> e.sw >> e.sh; entries.push_back(e); }
        else if (key == "body") s >> body;
    }
    if (L.width <= 0 || L.height <= 0 || entries.empty()) throw std::runtime_error("incomplete bev_lut.txt");
    const size_t n = size_t(L.width) * L.height;
    for (const Entry& e : entries) {
        Camera c;
        c.tag = e.tag;
        c.sensor_w = e.sw;
        c.sensor_h = e.sh;
        c.map = read_array<float>(dir + "/" + e.map, n * 2);
        c.weight = read_array<uint8_t>(dir + "/" + e.weight, n);
        L.cams.push_back(std::move(c));
    }
    L.body = body.empty() ? std::vector<uint8_t>(n, 0) : read_array<uint8_t>(dir + "/" + body, n);
    return L;
}

inline Image read_pnm(const std::string& path) {
    std::vector<uint8_t> b = read_all(path);
    size_t i = 0;
    auto token = [&]() {
        while (i < b.size()) {
            if (b[i] == '#') { while (i < b.size() && b[i] != '\n') ++i; }
            else if (std::isspace(b[i])) ++i;
            else break;
        }
        std::string t;
        while (i < b.size() && !std::isspace(b[i])) t += char(b[i++]);
        return t;
    };
    std::string magic = token();
    if (magic != "P5" && magic != "P6") throw std::runtime_error(path + ": not a binary PGM/PPM");
    Image im;
    im.c = magic == "P5" ? 1 : 3;
    im.w = std::stoi(token());
    im.h = std::stoi(token());
    if (std::stoi(token()) != 255) throw std::runtime_error(path + ": only 8-bit images");
    ++i;                                   // the single whitespace after maxval
    const size_t n = size_t(im.w) * im.h * im.c;
    if (b.size() < i + n) throw std::runtime_error(path + ": truncated");
    im.px.assign(b.begin() + i, b.begin() + i + n);
    return im;
}

inline void write_ppm(const std::string& path, const Image& im) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << "P6\n" << im.w << " " << im.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(im.px.data()), std::streamsize(im.px.size()));
}

// Bilinear sample at OpenCV pixel coordinates (pixel centres on integers), outside pixels count as 0,
// rounded to whole grey levels as cv::remap does for 8-bit images.
static inline void sample(const Image& im, float x, float y, float out[3]) {
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const float fx = x - x0, fy = y - y0;
    float acc[3] = {0.f, 0.f, 0.f};
    for (int dy = 0; dy < 2; ++dy) {
        const int yy = y0 + dy;
        if (yy < 0 || yy >= im.h) continue;
        for (int dx = 0; dx < 2; ++dx) {
            const int xx = x0 + dx;
            if (xx < 0 || xx >= im.w) continue;
            const float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy);
            const uint8_t* p = &im.px[(size_t(yy) * im.w + xx) * im.c];
            for (int k = 0; k < 3; ++k) acc[k] += w * p[im.c == 1 ? 0 : k];
        }
    }
    for (int k = 0; k < 3; ++k) out[k] = std::round(acc[k]);
}

static void check_frames(const Lut& L, const std::vector<Image>& frames) {
    if (frames.size() != L.cams.size())
        throw std::runtime_error(std::to_string(frames.size()) + " frames for " + std::to_string(L.cams.size()) + " cameras");
}

// Per-camera, per-channel gains that make overlapping cameras agree (Brown & Lowe gain compensation):
// minimise sum N_ab ((g_a I_ab - g_b I_ba)^2 / sigma_n^2 + (1 - g)^2 / sigma_g^2) over well exposed shared ground.
inline Gains estimate_gains(const Lut& L, const std::vector<Image>& frames, float sigma_n = 10.f, float sigma_g = 0.5f,
                            int step = 2) {
    check_frames(L, frames);
    const int nc = int(L.cams.size());
    std::vector<double> n_all(nc * nc, 0), n(nc * nc, 0), sa(nc * nc * 3, 0), sb(nc * nc * 3, 0);
    std::vector<float> s(nc * 3);
    std::vector<char> ok(nc);
    for (int y = 0; y < L.height; y += step) {
        for (int x = 0; x < L.width; x += step) {
            const size_t p = size_t(y) * L.width + x;
            if (L.body[p]) continue;
            for (int c = 0; c < nc; ++c) {
                const Camera& cam = L.cams[c];
                ok[c] = cam.map[2 * p] >= 0.f;
                if (ok[c]) {
                    const float sx = float(frames[c].w) / cam.sensor_w, sy = float(frames[c].h) / cam.sensor_h;
                    sample(frames[c], cam.map[2 * p] * sx - 0.5f, cam.map[2 * p + 1] * sy - 0.5f, &s[3 * c]);
                }
            }
            for (int a = 0; a < nc; ++a) {
                if (!ok[a]) continue;
                for (int b = a + 1; b < nc; ++b) {
                    if (!ok[b]) continue;
                    n_all[a * nc + b] += 1;
                    const float la = (s[3 * a] + s[3 * a + 1] + s[3 * a + 2]) / 3.f;
                    const float lb = (s[3 * b] + s[3 * b + 1] + s[3 * b + 2]) / 3.f;
                    if (la <= 8 || la >= 247 || lb <= 8 || lb >= 247) continue;
                    n[a * nc + b] += 1;
                    for (int k = 0; k < 3; ++k) {
                        sa[(a * nc + b) * 3 + k] += s[3 * a + k];
                        sb[(a * nc + b) * 3 + k] += s[3 * b + k];
                    }
                }
            }
        }
    }
    Gains g(nc, {1.f, 1.f, 1.f});
    const double in2 = 1.0 / (double(sigma_n) * sigma_n), ig2 = 1.0 / (double(sigma_g) * sigma_g);
    for (int k = 0; k < 3; ++k) {
        std::vector<double> A(nc * nc, 0), rhs(nc, 0);
        for (int a = 0; a < nc; ++a) {
            for (int b = a + 1; b < nc; ++b) {
                const double N = n[a * nc + b];
                if (n_all[a * nc + b] < 50 || N < 50) continue;
                const double Ia = sa[(a * nc + b) * 3 + k] / N, Ib = sb[(a * nc + b) * 3 + k] / N;
                A[a * nc + a] += N * (Ia * Ia * in2 + ig2);
                A[b * nc + b] += N * (Ib * Ib * in2 + ig2);
                A[a * nc + b] -= N * Ia * Ib * in2;
                A[b * nc + a] -= N * Ia * Ib * in2;
                rhs[a] += N * ig2;
                rhs[b] += N * ig2;
            }
        }
        for (int i = 0; i < nc; ++i)
            if (A[i * nc + i] == 0) { A[i * nc + i] = 1; rhs[i] = 1; }
        // Gaussian elimination with partial pivoting (nc <= 8)
        for (int col = 0; col < nc; ++col) {
            int piv = col;
            for (int r = col + 1; r < nc; ++r)
                if (std::fabs(A[r * nc + col]) > std::fabs(A[piv * nc + col])) piv = r;
            for (int j = 0; j < nc; ++j) std::swap(A[col * nc + j], A[piv * nc + j]);
            std::swap(rhs[col], rhs[piv]);
            for (int r = col + 1; r < nc; ++r) {
                const double f = A[r * nc + col] / A[col * nc + col];
                for (int j = col; j < nc; ++j) A[r * nc + j] -= f * A[col * nc + j];
                rhs[r] -= f * rhs[col];
            }
        }
        std::vector<double> x(nc);
        for (int i = nc - 1; i >= 0; --i) {
            double v = rhs[i];
            for (int j = i + 1; j < nc; ++j) v -= A[i * nc + j] * x[j];
            x[i] = v / A[i * nc + i];
        }
        for (int i = 0; i < nc; ++i) g[i][k] = float(std::min(2.5, std::max(0.4, x[i])));
    }
    return g;
}

// out = sum_c weight_c * gain_c * frame_c(map_c) / sum_c weight_c; body and uncovered ground get flat colours.
inline void stitch(const Lut& L, const std::vector<Image>& frames, const Gains* gains, Image& out) {
    check_frames(L, frames);
    out.w = L.width;
    out.h = L.height;
    out.c = 3;
    out.px.assign(size_t(out.w) * out.h * 3, 0);
    const int nc = int(L.cams.size());
    std::vector<float> sx(nc), sy(nc);
    for (int c = 0; c < nc; ++c) {
        sx[c] = float(frames[c].w) / L.cams[c].sensor_w;
        sy[c] = float(frames[c].h) / L.cams[c].sensor_h;
    }
    const long long n = (long long)L.width * L.height;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long p = 0; p < n; ++p) {
        uint8_t* o = &out.px[size_t(p) * 3];
        if (L.body[p]) { std::memcpy(o, BODY_RGB, 3); continue; }
        float acc[3] = {0.f, 0.f, 0.f};
        int wsum = 0;
        for (int c = 0; c < nc; ++c) {
            const Camera& cam = L.cams[c];
            const int w = cam.weight[p];
            const float mx = cam.map[2 * p], my = cam.map[2 * p + 1];
            if (!w || mx < 0.f) continue;
            float s[3];
            sample(frames[c], mx * sx[c] - 0.5f, my * sy[c] - 0.5f, s);
            for (int k = 0; k < 3; ++k) acc[k] += w * s[k] * (gains ? (*gains)[c][k] : 1.f);
            wsum += w;
        }
        if (!wsum) { std::memcpy(o, EMPTY_RGB, 3); continue; }
        for (int k = 0; k < 3; ++k) o[k] = uint8_t(std::min(255.f, std::max(0.f, acc[k] / wsum)));
    }
}

}  // namespace vbev

#ifndef VBEV_NO_MAIN
static bool exists(const std::string& p) { std::ifstream f(p); return bool(f); }

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    std::string frames_dir;
    bool balance = false;
    int bench = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--balance") balance = true;
        else if (a == "--bench" && i + 1 < argc) bench = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) frames_dir = argv[++i];
        else if (a == "-h" || a == "--help") { pos.clear(); break; }
        else pos.push_back(a);
    }
    if (pos.size() < 2) {
        std::cerr << "usage: bev_stitch LUT_DIR OUT.ppm (FRAME... | --frames DIR) [--balance] [--bench N]\n"
                     "  frames: binary PGM or PPM, one per camera in bev_lut.txt order (or DIR/<tag>.ppm|.pgm)\n";
        return 2;
    }
    try {
        const vbev::Lut L = vbev::load(pos[0]);
        std::vector<vbev::Image> frames;
        if (!frames_dir.empty()) {
            for (const auto& c : L.cams) {
                std::string p = frames_dir + "/" + c.tag + ".ppm";
                if (!exists(p)) p = frames_dir + "/" + c.tag + ".pgm";
                frames.push_back(vbev::read_pnm(p));
            }
        } else {
            for (size_t i = 2; i < pos.size(); ++i) frames.push_back(vbev::read_pnm(pos[i]));
        }
        vbev::Gains g;
        if (balance) {
            g = vbev::estimate_gains(L, frames);
            for (size_t c = 0; c < g.size(); ++c)
                std::printf("gain %-6s %.4f %.4f %.4f\n", L.cams[c].tag.c_str(), g[c][0], g[c][1], g[c][2]);
        }
        vbev::Image out;
        vbev::stitch(L, frames, balance ? &g : nullptr, out);
        if (bench > 0) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < bench; ++i) vbev::stitch(L, frames, balance ? &g : nullptr, out);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / bench;
            std::printf("stitch %d x %d from %zu cameras: %.2f ms per frame\n", L.width, L.height, L.cams.size(), ms);
        }
        vbev::write_ppm(pos[1], out);
        std::printf("wrote %s (%d x %d)\n", pos[1].c_str(), out.w, out.h);
    } catch (const std::exception& e) {
        std::cerr << "bev_stitch: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
#endif
