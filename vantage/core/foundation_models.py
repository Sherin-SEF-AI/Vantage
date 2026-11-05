"""Foundation models on rendered frames: open-vocabulary 2D detectors and monocular depth models, downloaded on request
from a pinned revision of their Hugging Face repositories, checked file by file against their published size and
SHA-256, and run through PyTorch and Hugging Face Transformers when they are installed (optional imports: the app's
downloads leave torch out, and every entry point says so instead of failing).

The models (licences checked on 2026-11-05 against each repository's model card and, where the card says nothing, its
source repository; the revision is the commit the files are pinned to)
-------------------------------------------------------------------------------------------------------------------
* OWLv2 base/16 ensemble (google/owlv2-base-patch16-ensemble, revision cfd3195b; Apache-2.0: the model card's licence
  field). Minderer, Gritsenko and Houlsby, "Scaling Open-Vocabulary Object Detection", NeurIPS 2023, arXiv:2306.09683.
  A CLIP image and text tower with box and class heads: every image patch gives a box and a similarity to each text
  query. The image is padded to a square (grey, bottom and right) and resized to 960 × 960.
* Grounding DINO tiny (IDEA-Research/grounding-dino-tiny, revision a2bb814d; Apache-2.0: the model card's licence
  field). Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection",
  arXiv:2303.05499. Swin-T backbone, BERT text encoder; the queries are joined into one prompt ("a car. a child.").
  Published figure: COCO zero-shot AP 48.4 for GroundingDINO-T (the GroundingDINO README's model-zoo table,
  github.com/IDEA-Research/GroundingDINO), not measured here.
* Depth Anything V2 Small, relative (depth-anything/Depth-Anything-V2-Small-hf, revision 5426e4f0; Apache-2.0: the model
  card's licence field, and the Depth-Anything-V2 README's LICENSE section: "Depth-Anything-V2-Small model is under the
  Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license"). Yang et al.,
  "Depth Anything V2", arXiv:2406.09414. Gives affine-invariant inverse depth (larger is nearer): Vantage turns it
  into metres per frame from the ground plane (core.foundation_range: the camera's height and pose say how far the
  ground is at every pixel below the horizon).
* Depth Anything V2 Small, metric outdoor (depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf, revision
  fd2c2202): planar depth in metres (up to 80 m), fine-tuned on Virtual KITTI 2. Its Transformers card declares no
  licence; the checkpoint it converts (depth-anything/Depth-Anything-V2-Metric-VKITTI-Small) declares Apache-2.0, but
  the fine-tuning data, Virtual KITTI 2, is published under CC BY-NC-SA 3.0. Offered only when the user opts in to
  models whose commercial use is not clear, and the page says why.
* Depth Anything V2 Base, relative (depth-anything/Depth-Anything-V2-Base-hf, revision b1958afc; CC-BY-NC-4.0: the
  model card's licence field): non-commercial use only; offered only on the same opt-in.
YOLOX-S (core.detection, Apache-2.0, through OpenCV) is the closed-vocabulary baseline the study runs beside them.

Running
-------
On an NVIDIA GPU when torch sees one with enough free memory (its share capped at gpu_mem_gb), else on the CPU (at
most 8 threads). VANTAGE_GPU_LOCK, when set, names a lock file held (flock) while a model runs on the GPU, so jobs that
share a GPU take turns. float32 throughout. Inference speeds are measured, not assumed (Detector.ms per image).
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import detection as DT

SCHEMA = 'vantage.foundation_models/1'
ENV_DIR = 'VANTAGE_FOUNDATION_DIR'          # overrides the folder (tests, shared installs)
ENV_GPU_LOCK = 'VANTAGE_GPU_LOCK'
TASKS = ('detect', 'depth')
HF = 'https://huggingface.co/{repo}/resolve/{rev}/{file}'
TORCH_MISSING = ('The foundation-model study needs torch and transformers (PyTorch and Hugging Face Transformers), '
                 'which the downloads do not include: run Vantage from source with "pip install torch transformers" '
                 '(the CPU build of torch is enough; an NVIDIA GPU is much faster).')


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class FoundationSpec:
    id: str
    name: str
    task: str                    # detect | depth
    arch: str                    # owlv2 | grounding_dino | depth_anything
    repo: str
    revision: str
    files: tuple
    licence: str
    licence_url: str
    commercial: bool             # the licence allows commercial use (False: offered only on the user's opt-in)
    licence_note: str
    paper: str
    params_m: float
    depth_kind: str = ''         # depth models: 'metric' (metres) | 'relative' (inverse depth up to scale and shift)
    max_depth_m: float = 0.0
    note: str = ''

    @property
    def size(self) -> int:
        return int(sum(f.size for f in self.files))

    def to_dict(self):
        d = asdict(self)
        d['files'] = [asdict(f) for f in self.files]
        d['size'] = self.size
        return d


def _f(*rows):
    return tuple(FileSpec(n, s, h) for n, s, h in rows)


SPECS = {
    'owlv2_b16_ens': FoundationSpec(
        'owlv2_b16_ens', 'OWLv2 B/16 ensemble', 'detect', 'owlv2', 'google/owlv2-base-patch16-ensemble',
        'cfd3195ba4ea9592eec887ded089f4c08eff231d',
        _f(('config.json', 414, 'ba9df8c25a4b8461887dd0a93d9252c9cd84697fe8d49a9d8794ce409af9acb2'),
           ('preprocessor_config.json', 425, 'cf3e396635b797ee1a464e1b2836e98748f8edac19e89aaa2c93b55ac15b0064'),
           ('tokenizer_config.json', 1100, 'b55cda6198e152ded427c8a9b3faf1cccf27a7fa080697a62f6ff143f511f44f'),
           ('special_tokens_map.json', 121, 'd6e2b9cf664efbad2d22998b8d3da986abcbeed3e0825ad33605c9401f9cf73e'),
           ('added_tokens.json', 67, 'e5dc0da35d20111e8ff3fdfc03682beca23d5f94ed74331bce81786b2636a24f'),
           ('vocab.json', 1059962, 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349'),
           ('merges.txt', 524619, '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a'),
           ('model.safetensors', 619918824, 'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7')),
        'Apache-2.0', 'https://huggingface.co/google/owlv2-base-patch16-ensemble', True,
        'Apache-2.0 (the model card\'s licence field)',
        'Minderer, Gritsenko, Houlsby: Scaling Open-Vocabulary Object Detection, NeurIPS 2023, arXiv:2306.09683', 155.0,
        note='text queries scored against every image patch; 960 × 960 input'),
    'gdino_tiny': FoundationSpec(
        'gdino_tiny', 'Grounding DINO tiny', 'detect', 'grounding_dino', 'IDEA-Research/grounding-dino-tiny',
        'a2bb814dd30d776dcf7e30523b00659f4f141c71',
        _f(('config.json', 1644, 'eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a'),
           ('preprocessor_config.json', 457, '8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60'),
           ('tokenizer_config.json', 1237, 'd40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56'),
           ('special_tokens_map.json', 125, 'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3'),
           ('added_tokens.json', 82, '909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2'),
           ('tokenizer.json', 711396, 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'),
           ('vocab.txt', 231508, '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
           ('model.safetensors', 689359096, '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3')),
        'Apache-2.0', 'https://huggingface.co/IDEA-Research/grounding-dino-tiny', True,
        'Apache-2.0 (the model card\'s licence field)',
        'Liu et al.: Grounding DINO, arXiv:2303.05499 (COCO zero-shot AP 48.4 for GroundingDINO-T: the GroundingDINO '
        'README\'s model-zoo table)', 172.0, note='one prompt of every query; 800 px short side'),
    'da2_small': FoundationSpec(
        'da2_small', 'Depth Anything V2 Small (relative)', 'depth', 'depth_anything',
        'depth-anything/Depth-Anything-V2-Small-hf', '5426e4f0f36572d16453bbda7a8389317b1bef99',
        _f(('config.json', 950, 'c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75'),
           ('preprocessor_config.json', 775, 'd41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84'),
           ('model.safetensors', 99173660, '3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70')),
        'Apache-2.0', 'https://github.com/DepthAnything/Depth-Anything-V2#license', True,
        'Apache-2.0 (the model card\'s licence field; the Depth-Anything-V2 README: "Depth-Anything-V2-Small model is '
        'under the Apache-2.0 license")',
        'Yang et al.: Depth Anything V2, arXiv:2406.09414', 24.8, depth_kind='relative',
        note='inverse depth up to scale and shift: scaled per frame from the ground plane'),
    'da2_metric_outdoor_small': FoundationSpec(
        'da2_metric_outdoor_small', 'Depth Anything V2 Small (metric, outdoor)', 'depth', 'depth_anything',
        'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf', 'fd2c22027eaf20374204f14099b8341e1925ad39',
        _f(('config.json', 1063, '3417ccb9df6fafe7e03c46583db7185d15e8ad4f5bbbdd28295bb06cfb132b48'),
           ('preprocessor_config.json', 437, '533b16a60445d7cab5086d39b45f92be45624b977972c62d5984d93e98366063'),
           ('model.safetensors', 99173660, 'ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124')),
        'Apache-2.0 (checkpoint); trained on CC BY-NC-SA 3.0 data',
        'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small', False,
        'Its Transformers card declares no licence; the checkpoint it converts (Depth-Anything-V2-Metric-VKITTI-Small) '
        'declares Apache-2.0, but it is fine-tuned on Virtual KITTI 2, published under CC BY-NC-SA 3.0 (non-'
        'commercial): whether a commercial use is allowed is not clear. Offered only when you opt in.',
        'Yang et al.: Depth Anything V2, arXiv:2406.09414 (metric depth: its metric_depth README)', 24.8,
        depth_kind='metric', max_depth_m=80.0, note='planar depth in metres, at most 80 m (Virtual KITTI 2)'),
    'da2_base': FoundationSpec(
        'da2_base', 'Depth Anything V2 Base (relative)', 'depth', 'depth_anything',
        'depth-anything/Depth-Anything-V2-Base-hf', 'b1958afc87fb45a9e3746cb387596094de553ed8',
        _f(('config.json', 953, '1f616484953fbbddcc14da69ab81ef834fd58c11be04fa2f2eecf5696b2f7389'),
           ('preprocessor_config.json', 775, 'd41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84'),
           ('model.safetensors', 389916980, '9996d15c334d264cfcc29f4318d7d3789a006ecc444782c2fb82f7eaa1acb8bf')),
        'CC-BY-NC-4.0', 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf', False,
        'CC-BY-NC-4.0 (the model card\'s licence field): non-commercial use only. Offered only when you opt in.',
        'Yang et al.: Depth Anything V2, arXiv:2406.09414', 97.5, depth_kind='relative',
        note='larger and slower than Small; inverse depth scaled from the ground plane'),
}
DETECTORS = tuple(k for k, s in SPECS.items() if s.task == 'detect')
DEPTH_MODELS = tuple(k for k, s in SPECS.items() if s.task == 'depth')
BASELINES = tuple(DT.MODELS)                  # YOLOX through OpenCV (core.detection)
DEFAULT_DETECTORS = ('yolox_s', 'owlv2_b16_ens')
DEFAULT_DEPTH = 'da2_small'


def torch_missing() -> str | None:
    """Why the models cannot run here (no torch or no transformers), or None."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return TORCH_MISSING
    return None


def spec_of(model) -> FoundationSpec:
    if isinstance(model, FoundationSpec):
        return model
    if model not in SPECS:
        raise KeyError(f'Unknown foundation model {model!r}; known: {", ".join(SPECS)}')
    return SPECS[model]


def allowed(model, opt_in: bool = False) -> bool:
    """A model the study may use: its licence allows commercial use, or the user opted in."""
    sp = spec_of(model)
    return bool(sp.commercial or opt_in)


def model_label(mid: str) -> str:
    if mid in SPECS:
        return SPECS[mid].name
    if mid in DT.MODELS:
        return DT.MODELS[mid].name
    return str(mid)


# =========================================================================== where models live
def foundation_dir() -> str:
    """$VANTAGE_FOUNDATION_DIR, else the folder next to core.detection's models folder: <data folder>/foundation."""
    env = os.environ.get(ENV_DIR)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.dirname(DT.models_dir()), 'foundation')


def model_dir(model, folder=None) -> str:
    return os.path.join(folder or foundation_dir(), spec_of(model).id)


def status(model, folder=None, verify=True) -> dict:
    """{'id', 'name', 'path', 'present', 'verified', 'size', 'expected_size', 'licence', 'commercial', 'text',
    'files': [{'name', 'present', 'verified'}]}: every file there with its published size and SHA-256 (a wrong file
    is never used). verify False checks the sizes only (a quick look for a list; loading always checks the SHA-256,
    which is cached per file once computed)."""
    sp = spec_of(model)
    d = model_dir(sp, folder)
    rows, have, ok_all = [], 0, True
    for f in sp.files:
        p = os.path.join(d, f.name)
        r = {'name': f.name, 'present': os.path.isfile(p), 'verified': False}
        if r['present']:
            n = os.path.getsize(p)
            have += n
            if n == f.size:
                try:
                    r['verified'] = (DT.file_sha256(p) == f.sha256) if verify else True
                except OSError:
                    r['verified'] = False
        ok_all &= r['verified']
        rows.append(r)
    present = any(r['present'] for r in rows)
    out = {'id': sp.id, 'name': sp.name, 'task': sp.task, 'path': d, 'present': present, 'verified': ok_all,
           'size': have, 'expected_size': sp.size, 'licence': sp.licence, 'licence_url': sp.licence_url,
           'commercial': sp.commercial, 'files': rows, 'repo': sp.repo, 'revision': sp.revision}
    mb = DT._mb(sp.size)
    out['checked'] = 'sha256' if verify else 'size'
    if ok_all:
        out['text'] = (f'{sp.name} ready ({mb}, every file SHA-256 checked, {sp.licence}).' if verify else
                       f'{sp.name} downloaded ({mb}, sizes match; the SHA-256 is checked when it loads; '
                       f'{sp.licence}).')
    elif not present:
        out['text'] = f'{sp.name} is not downloaded ({mb}, {sp.licence}).'
    else:
        bad = [r['name'] for r in rows if not r['verified']]
        out['text'] = (f'{sp.name}: {len(bad)} of {len(rows)} files missing or not the published ones '
                       f'({", ".join(bad[:3])}{"…" if len(bad) > 3 else ""}): download it again.')
    return out


class _Cancelled(Exception):
    pass


def download(model, progress=None, cancel=None, folder=None, timeout=60.0, opt_in=False, fetch=None) -> str | None:
    """Download every file of a model from its pinned revision into <folder>/<id>; returns the folder (None when
    cancelled). Files already there with the published size and SHA-256 are kept; each new file is written as .part
    and renamed into place only when its size and SHA-256 match; a mismatch raises ValueError (and nothing of that
    file is kept), the network OSError. A model whose licence does not allow commercial use needs opt_in.
    fetch(url, timeout) -> a readable response (tests: a fake server)."""
    import urllib.request
    from .. import __version__
    sp = spec_of(model)
    if not allowed(sp, opt_in):
        raise PermissionError(f'{sp.name}: {sp.licence_note}')
    d = model_dir(sp, folder)
    os.makedirs(d, exist_ok=True)
    stop = cancel or (lambda: False)
    total = sp.size
    done = 0
    t0 = time.perf_counter()

    def opener(url):
        if fetch is not None:
            return fetch(url, timeout)
        req = urllib.request.Request(url, headers={'User-Agent': f'Vantage/{__version__}'})
        kw = {'context': DT._ssl_context()} if url.lower().startswith('https:') else {}
        return urllib.request.urlopen(req, timeout=timeout, **kw)
    for f in sp.files:
        final = os.path.join(d, f.name)
        if os.path.isfile(final) and os.path.getsize(final) == f.size and DT.file_sha256(final) == f.sha256:
            done += f.size
            continue
        part = final + '.part'
        url = HF.format(repo=sp.repo, rev=sp.revision, file=f.name)
        h = hashlib.sha256()
        n = 0
        try:
            with opener(url) as r, open(part, 'wb') as fh:
                while True:
                    if stop():
                        raise _Cancelled()
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    h.update(block)
                    n += len(block)
                    if n > f.size * 1.01 + (1 << 20):
                        raise ValueError(f'{sp.name}: {f.name} is larger than the published {f.size:,} bytes.')
                    if progress is not None:
                        dt = max(time.perf_counter() - t0, 1e-3)
                        progress(min((done + n) / max(total, 1), 1.0),
                                 f'{f.name} · {DT._mb(done + n)} of {DT._mb(total)} · {(done + n) / dt / 1e6:.1f} MB/s')
            if n != f.size:
                raise ValueError(f'{sp.name}: {f.name} has {n:,} bytes, the published file {f.size:,}.')
            if h.hexdigest() != f.sha256:
                raise ValueError(f'{sp.name}: {f.name}\'s SHA-256 ({h.hexdigest()[:16]}…) is not the published '
                                 f'file\'s ({f.sha256[:16]}…): not kept.')
            os.replace(part, final)
        except _Cancelled:
            DT._rm(part)
            return None
        except BaseException:
            DT._rm(part)
            raise
        done += f.size
        with DT._SHA_LOCK:
            st = os.stat(final)
            DT._SHA_CACHE[(os.path.abspath(final), st.st_size, st.st_mtime_ns)] = f.sha256
    if progress is not None:
        progress(1.0, f'{sp.name} ready')
    return d


def remove(model, folder=None) -> bool:
    d = model_dir(model, folder)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False


# =========================================================================== the device
def device_info(pref: str = 'auto', mem_gb: float = 6.0) -> dict:
    """{'device': 'cuda' | 'cpu', 'name', 'free_gb', 'text'} without loading a model (None of torch: 'none')."""
    try:
        import torch
    except ImportError:
        return {'device': 'none', 'name': '', 'free_gb': None, 'text': 'No torch: the foundation models cannot run.'}
    if pref in ('auto', 'cuda') and torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info(0)
            name = torch.cuda.get_device_name(0)
        except RuntimeError:                              # pragma: no cover — a busy or broken driver
            free, total, name = 0, 0, 'CUDA'
        if free >= min(mem_gb, 2.0) * 1e9:
            return {'device': 'cuda', 'name': name, 'free_gb': free / 1e9, 'total_gb': total / 1e9,
                    'text': f'GPU: {name} ({free / 1e9:.1f} of {total / 1e9:.1f} GB free)'}
        return {'device': 'cpu', 'name': 'CPU', 'free_gb': free / 1e9,
                'text': f'CPU (the GPU {name} has only {free / 1e9:.1f} GB free)'}
    return {'device': 'cpu', 'name': 'CPU', 'free_gb': None,
            'text': 'CPU (no CUDA GPU)' if pref != 'cpu' else 'CPU (chosen)'}


def _torch_device(pref: str, mem_gb: float):
    import torch
    info = device_info(pref, mem_gb)
    if info['device'] == 'cuda':
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(max(mem_gb * 1e9 / total, 0.05), 0.9), 0)
        except (RuntimeError, AssertionError):            # pragma: no cover
            pass
        return torch.device('cuda', 0), info
    n = max(1, min(8, (os.cpu_count() or 2) // 2))
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)
    return torch.device('cpu'), info


@contextlib.contextmanager
def gpu_lock(device: str):
    """Hold $VANTAGE_GPU_LOCK (flock) while a model runs on the GPU; nothing otherwise (or where flock is missing)."""
    path = os.environ.get(ENV_GPU_LOCK)
    if device != 'cuda' or not path:
        yield
        return
    try:
        import fcntl
    except ImportError:                                   # pragma: no cover — Windows
        yield
        return
    with open(path, 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load_check(sp: FoundationSpec, folder=None) -> str:
    why = torch_missing()
    if why:
        raise ValueError(why)
    try:                                                  # no progress bars or advice on the console
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:                                     # noqa: BLE001 — an older transformers
        pass
    s = status(sp, folder)
    if not s['present']:
        raise FileNotFoundError(f'{sp.name} is not downloaded: Simulate ▸ Foundation models ▸ Download '
                                f'({DT._mb(sp.size)}).')
    if not s['verified']:
        raise ValueError(s['text'])
    return s['path']


def _rgb(img):
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, 2)
    return np.ascontiguousarray(a[:, :, 2::-1])           # BGR (OpenCV) -> RGB


# =========================================================================== open-vocabulary detection
@dataclass
class DetectorOptions:
    keep_score: float = 0.05          # detections kept (the study's operating point is higher)
    text_score: float = 0.25          # Grounding DINO: a token's score for the box to carry its phrase
    max_dets: int = 100
    nms_iou: float = 0.5              # per query (OWLv2 gives one box per patch and query)
    device: str = 'auto'              # auto | cuda | cpu
    gpu_mem_gb: float = 6.0


class OpenVocabDetector:
    """One loaded open-vocabulary detector with a fixed list of text queries. detect(bgr) -> {'boxes' n × 4 (x0 y0 x1
    y1, image px), 'scores', 'labels' (index into queries), 'ms'}."""

    def __init__(self, model, queries, options: DetectorOptions | None = None, folder=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.spec = spec_of(model)
        if self.spec.task != 'detect':
            raise ValueError(f'{self.spec.name} is not a detector.')
        self.queries = [str(q).strip() for q in queries if str(q).strip()]
        if not self.queries:
            raise ValueError('No text queries to detect.')
        self.opt = options or DetectorOptions()
        path = _load_check(self.spec, folder)
        self.device, self.info = _torch_device(self.opt.device, self.opt.gpu_mem_gb)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True)
        self.model.eval().to(self.device)
        self.torch = torch
        self.lock = threading.Lock()
        if self.spec.arch == 'grounding_dino':
            self.prompt = ' '.join(q.rstrip('.').lower() + '.' for q in self.queries)
            self._words = [set(_words(q)) for q in self.queries]

    def describe(self) -> dict:
        return {'model': self.spec.id, 'name': self.spec.name, 'repo': self.spec.repo, 'revision': self.spec.revision,
                'licence': self.spec.licence, 'device': self.info.get('text'), 'queries': list(self.queries),
                'options': asdict(self.opt)}

    def _label_of(self, text) -> int:
        """Grounding DINO's phrase -> the query it came from (the most shared words; -1 for none)."""
        w = set(_words(text))
        best, bi = 0, -1
        for i, qw in enumerate(self._words):
            k = len(w & qw - {'a', 'an', 'the'})
            if k > best:
                best, bi = k, i
        return bi

    def detect(self, img) -> dict:
        torch = self.torch
        t0 = time.perf_counter()
        rgb = _rgb(img)
        h, w = rgb.shape[:2]
        with self.lock, torch.inference_mode(), gpu_lock(self.info['device']):
            if self.spec.arch == 'owlv2':
                inp = self.processor(text=[self.queries], images=[rgb], return_tensors='pt').to(self.device)
                out = self.model(**inp)
                side = max(h, w)                  # OWLv2 pads to a square at the bottom and right
                logits = out.logits[0].sigmoid()  # (patches, queries)
                sc, lab = logits.max(-1)
                bx = out.pred_boxes[0]            # cx cy w h relative to the padded square
                keep = sc > self.opt.keep_score
                sc, lab, bx = sc[keep], lab[keep], bx[keep]
                b = torch.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2,
                                 bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], 1) * side
                boxes, scores, labels = (b.float().cpu().numpy(), sc.float().cpu().numpy(),
                                         lab.cpu().numpy().astype(int))
            else:
                inp = self.processor(images=[rgb], text=[self.prompt], return_tensors='pt').to(self.device)
                out = self.model(**inp)
                res = self.processor.post_process_grounded_object_detection(
                    out, inp['input_ids'], threshold=self.opt.keep_score, text_threshold=self.opt.text_score,
                    target_sizes=[(h, w)])[0]
                boxes = res['boxes'].float().cpu().numpy()
                scores = res['scores'].float().cpu().numpy()
                texts = res.get('text_labels', res.get('labels'))
                labels = np.array([self._label_of(t) for t in texts], int) if len(scores) else np.zeros(0, int)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        ok = (labels >= 0) & (boxes[:, 2] - boxes[:, 0] > 0.5) & (boxes[:, 3] - boxes[:, 1] > 0.5)
        boxes, scores, labels = boxes[ok], np.asarray(scores, np.float32)[ok], labels[ok]
        keep = _nms_per_label(boxes, scores, labels, self.opt.nms_iou)
        order = keep[np.argsort(-scores[keep], kind='mergesort')][:self.opt.max_dets]
        return {'boxes': boxes[order], 'scores': scores[order], 'labels': labels[order],
                'ms': (time.perf_counter() - t0) * 1000.0}

    __call__ = detect

    def close(self):
        """Free the model (and the GPU memory it holds)."""
        self.model = None
        try:
            if self.info.get('device') == 'cuda':
                self.torch.cuda.empty_cache()
        except Exception:                                  # noqa: BLE001
            pass


def _words(s) -> list:
    return [w for w in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in str(s)).split() if w]


def _nms_per_label(boxes, scores, labels, thr) -> np.ndarray:
    """Indices kept by non-maximum suppression within each label (numpy; highest score first)."""
    from .detection_eval import iou_matrix
    keep = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        idx = idx[np.argsort(-scores[idx], kind='mergesort')]
        ious = iou_matrix(boxes[idx], boxes[idx])
        dead = np.zeros(len(idx), bool)
        for a in range(len(idx)):
            if dead[a]:
                continue
            keep.append(idx[a])
            dead |= ious[a] > thr
    return np.asarray(sorted(keep), int)


# =========================================================================== monocular depth
class DepthModel:
    """One loaded depth model. predict(bgr) -> float32 H × W at the image's size: metres (metric models) or relative
    inverse depth (relative models: larger is nearer, up to an unknown scale and shift)."""

    def __init__(self, model, device='auto', gpu_mem_gb=6.0, folder=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.spec = spec_of(model)
        if self.spec.task != 'depth':
            raise ValueError(f'{self.spec.name} is not a depth model.')
        path = _load_check(self.spec, folder)
        self.device, self.info = _torch_device(device, gpu_mem_gb)
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True)
        self.model.eval().to(self.device)
        self.torch = torch
        self.lock = threading.Lock()
        self.kind = self.spec.depth_kind

    def describe(self) -> dict:
        return {'model': self.spec.id, 'name': self.spec.name, 'repo': self.spec.repo, 'revision': self.spec.revision,
                'licence': self.spec.licence, 'kind': self.kind, 'device': self.info.get('text')}

    def predict(self, img) -> np.ndarray:
        torch = self.torch
        rgb = _rgb(img)
        h, w = rgb.shape[:2]
        with self.lock, torch.inference_mode(), gpu_lock(self.info['device']):
            inp = self.processor(images=rgb, return_tensors='pt').to(self.device)
            out = self.model(**inp).predicted_depth            # (1, h', w')
            d = torch.nn.functional.interpolate(out[:, None].float(), size=(h, w), mode='bicubic',
                                                align_corners=False)[0, 0]
            arr = d.cpu().numpy().astype(np.float32)
        if self.kind == 'metric':
            arr = np.maximum(arr, 0.0)
        return arr

    __call__ = predict

    def close(self):
        self.model = None
        try:
            if self.info.get('device') == 'cuda':
                self.torch.cuda.empty_cache()
        except Exception:                                  # noqa: BLE001
            pass


# =========================================================================== the baseline in the same form
class YoloxAsQueries:
    """YOLOX (core.detection) answering the study's class list: detect(bgr) -> {'boxes', 'scores', 'labels' (index
    into the classes: a COCO class mapped to a study class through its 'coco' name; classes YOLOX does not know get
    nothing), 'ms'}."""

    def __init__(self, coco_names, settings: DT.DetectorSettings | None = None, folder=None, detector=None):
        self.det = detector if detector is not None else DT.load(settings or DT.DetectorSettings(), folder)
        self.map = {}
        for i, c in enumerate(coco_names):             # the first study class of each COCO class answers it
            if c in DT.CLASSES and DT.CLASSES.index(c) not in self.map:
                self.map[DT.CLASSES.index(c)] = i
        spec = getattr(self.det, 'spec', None)
        self.spec = spec
        self.id = getattr(spec, 'id', 'yolox_s')

    def describe(self) -> dict:
        d = self.det.describe() if hasattr(self.det, 'describe') else {'model': self.id}
        d['classes_mapped'] = {DT.CLASSES[k]: v for k, v in self.map.items()}
        return d

    def detect(self, img) -> dict:
        t0 = time.perf_counter()
        d = self.det(img)
        cls = np.asarray(d['cls'], int)
        lab = np.array([self.map.get(int(c), -1) for c in cls], int)
        ok = lab >= 0
        return {'boxes': np.asarray(d['boxes'], np.float32).reshape(-1, 4)[ok],
                'scores': np.asarray(d['scores'], np.float32)[ok], 'labels': lab[ok],
                'ms': float(d.get('ms', (time.perf_counter() - t0) * 1000.0))}

    __call__ = detect

    def close(self):
        pass


def catalogue(opt_in: bool = False, folder=None, verify=False) -> list:
    """Every model the page lists: {'id', 'name', 'task', 'licence', 'commercial', 'allowed', 'size', 'status'}; YOLOX
    (core.detection's models) first. verify False: the foundation models' sizes only (quick)."""
    out = []
    for m in DT.MODELS.values():
        s = DT.status(m.id)
        out.append({'id': m.id, 'name': m.name, 'task': 'detect', 'arch': 'yolox', 'licence': m.licence,
                    'licence_note': 'Apache-2.0 (YOLOX\'s repository)', 'commercial': True, 'allowed': True,
                    'size': m.size, 'verified': s['verified'], 'present': s['present'], 'text': s['text'],
                    'repo': m.source, 'paper': 'Ge et al.: YOLOX, arXiv:2107.08430', 'closed_vocabulary': True})
    for sp in SPECS.values():
        s = status(sp, folder, verify)
        out.append({'id': sp.id, 'name': sp.name, 'task': sp.task, 'arch': sp.arch, 'licence': sp.licence,
                    'licence_note': sp.licence_note, 'commercial': sp.commercial, 'allowed': allowed(sp, opt_in),
                    'size': sp.size, 'verified': s['verified'], 'present': s['present'], 'text': s['text'],
                    'repo': f'https://huggingface.co/{sp.repo}/tree/{sp.revision}', 'paper': sp.paper,
                    'closed_vocabulary': False, 'depth_kind': sp.depth_kind})
    return out


__all__ = ['SPECS', 'FoundationSpec', 'FileSpec', 'DETECTORS', 'DEPTH_MODELS', 'torch_missing', 'status', 'download',
           'remove', 'foundation_dir', 'model_dir', 'OpenVocabDetector', 'DepthModel', 'YoloxAsQueries', 'catalogue',
           'device_info', 'DetectorOptions', 'allowed']
