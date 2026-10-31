// Vantage website: videos play while on screen (never for readers who prefer reduced motion), click to enlarge,
// mobile navigation, docs table of contents that follows the reading position, copy buttons on code blocks.
(function () {
  'use strict';
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- navigation
  var toggle = document.querySelector('.nav-toggle');
  var links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', function () {
      var open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    links.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') { links.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); }
    });
  }

  // ---------------------------------------------------------------- videos
  var frames = Array.prototype.slice.call(document.querySelectorAll('.frame'));
  frames.forEach(function (frame) {
    var v = frame.querySelector('video');
    if (!v) { return; }
    v.muted = true;
    v.setAttribute('muted', '');
    v.setAttribute('playsinline', '');
    var play = frame.querySelector('.play');
    if (play) {
      play.addEventListener('click', function () {
        if (v.paused) { v.play(); frame.classList.remove('paused'); frame.dataset.user = 'play'; }
        else { v.pause(); frame.classList.add('paused'); frame.dataset.user = 'pause'; }
      });
    }
    if (reduce) { frame.classList.add('paused'); }
  });
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var frame = en.target, v = frame.querySelector('video');
        if (!v) { return; }
        if (en.isIntersecting && en.intersectionRatio > 0.35) {
          if (frame.dataset.user === 'pause') { return; }
          if (v.preload === 'none') { v.preload = 'auto'; }
          var p = v.play();
          if (p && p.catch) { p.catch(function () { frame.classList.add('paused'); }); }
          frame.classList.remove('paused');
        } else if (!v.paused) {
          v.pause();
        }
      });
    }, { threshold: [0, 0.35, 0.6] });
    frames.forEach(function (f) { if (f.querySelector('video')) { io.observe(f); } });
  }

  // ---------------------------------------------------------------- YouTube: the player loads only when asked for
  Array.prototype.forEach.call(document.querySelectorAll('.frame.yt a[data-yt]'), function (a) {
    a.addEventListener('click', function (e) {
      if (e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) { return; }
      e.preventDefault();
      var f = document.createElement('iframe');
      f.src = 'https://www.youtube-nocookie.com/embed/' + a.dataset.yt + '?autoplay=1&rel=0';
      f.title = a.getAttribute('aria-label');
      f.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
      f.allowFullscreen = true;
      f.referrerPolicy = 'strict-origin-when-cross-origin';
      var chip = a.parentNode.querySelector('.chip');
      if (chip) { chip.remove(); }
      a.replaceWith(f);
    });
  });

  // ---------------------------------------------------------------- lightbox
  var box = document.createElement('div');
  box.className = 'lightbox';
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-modal', 'true');
  box.setAttribute('aria-label', 'Enlarged media');
  var close = document.createElement('button');
  close.type = 'button';
  close.textContent = 'Close  Esc';
  box.appendChild(close);
  document.body.appendChild(box);
  var last = null;
  function shut() {
    var m = box.querySelector('video, img');
    if (m) { if (m.pause) { m.pause(); } m.remove(); }
    box.classList.remove('open');
    document.body.style.overflow = '';
    if (last) { last.focus(); }
  }
  function open(frame, trigger) {
    var src = frame.querySelector('video, img');
    if (!src) { return; }
    var el;
    if (src.tagName === 'VIDEO') {
      el = document.createElement('video');
      el.src = src.currentSrc || src.querySelector('source').src;
      el.controls = true; el.loop = true; el.muted = true; el.playsInline = true; el.autoplay = !reduce;
      el.poster = src.poster;
      try { el.currentTime = src.currentTime; } catch (e) { /* not seekable yet */ }
    } else {
      el = document.createElement('img');
      el.src = src.src; el.alt = src.alt;
    }
    box.insertBefore(el, close);
    box.classList.add('open');
    document.body.style.overflow = 'hidden';
    last = trigger;
    close.focus();
  }
  frames.forEach(function (frame) {
    var b = frame.querySelector('.expand');
    if (b) { b.addEventListener('click', function () { open(frame, b); }); }
    var img = frame.querySelector('img');
    if (img && !frame.querySelector('video') && !frame.classList.contains('yt')) {
      img.style.cursor = 'zoom-in';
      img.addEventListener('click', function () { open(frame, img); });
    }
  });
  close.addEventListener('click', shut);
  box.addEventListener('click', function (e) { if (e.target === box) { shut(); } });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && box.classList.contains('open')) { shut(); } });

  // ---------------------------------------------------------------- docs: table of contents and copy buttons
  var toc = document.querySelector('.toc');
  if (toc && 'IntersectionObserver' in window) {
    var map = {};
    Array.prototype.forEach.call(toc.querySelectorAll('a[href^="#"]'), function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var heads = Array.prototype.filter.call(document.querySelectorAll('.doc h2[id], .doc h3[id]'), function (h) { return map[h.id]; });
    var current = null;
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          if (current) { current.classList.remove('active'); }
          current = map[en.target.id];
          current.classList.add('active');
        }
      });
    }, { rootMargin: '-15% 0px -75% 0px' });
    heads.forEach(function (h) { spy.observe(h); });
  }
  Array.prototype.forEach.call(document.querySelectorAll('.doc pre'), function (pre) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'copy';
    b.textContent = 'Copy';
    b.addEventListener('click', function () {
      var text = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text.trim()).then(function () {
          b.textContent = 'Copied';
          setTimeout(function () { b.textContent = 'Copy'; }, 1400);
        });
      }
    });
    pre.appendChild(b);
  });
})();
