(() => {
  const backdrop = document.querySelector('#sceneBackdrop');
  const video = document.querySelector('#sceneVideo');
  const canvas = document.querySelector('#sceneCanvas');
  const story = document.querySelector('#scrollStory');
  const context = canvas.getContext('2d', { alpha: false });
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const connection = navigator.connection;
  const totalFrames = 48;
  const frames = new Map();
  let active = false;
  let generation = 0;
  let target = 0;
  let smoothed = 0;
  let request = 0;
  let drawn = -1;
  let cacheReady = false;
  let viewWidth = innerWidth;
  let viewHeight = innerHeight;
  let scrollRange = 1;

  function measure() {
    viewWidth = innerWidth;
    viewHeight = innerHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(viewWidth * ratio);
    canvas.height = Math.round(viewHeight * ratio);
    if (context) context.imageSmoothingQuality = 'high';
    // The response can grow after generation without moving the scene's timeline.
    scrollRange = Math.max(1, story.offsetTop + story.offsetHeight - viewHeight);
    drawn = -1;
    updateTarget();
  }

  function draw() {
    if (!context || !cacheReady) return;
    const wanted = Math.round(smoothed * (totalFrames - 1));
    const available = [...frames.keys()];
    const index = available.reduce((best, n) =>
      Math.abs(n - wanted) < Math.abs(best - wanted) ? n : best, available[0]);
    if (index === undefined || index === drawn) return;
    const frame = frames.get(index);
    const scale = Math.max(canvas.width / frame.width, canvas.height / frame.height);
    const width = frame.width * scale;
    const height = frame.height * scale;
    // A portrait viewport prioritizes Krishna instead of cropping both faces.
    const horizontalFocus = viewWidth <= 767 ? 0.35 : 0.5;
    context.drawImage(frame, (canvas.width - width) * horizontalFocus, (canvas.height - height) / 2, width, height);
    drawn = index;
    canvas.dataset.frame = String(index);
    canvas.classList.add('is-ready');
    backdrop.classList.add('has-frames');
  }

  function seekFallback() {
    if (cacheReady || !Number.isFinite(video.duration) || video.seeking) return;
    const time = smoothed * Math.max(0, video.duration - 0.05);
    if (Math.abs(video.currentTime - time) > 0.04) video.currentTime = time;
  }

  function tick() {
    request = 0;
    if (!active || document.hidden) return;
    smoothed += (target - smoothed) * 0.12;
    if (Math.abs(target - smoothed) < 0.0005) smoothed = target;
    if (cacheReady) draw();
    else seekFallback();
    if (smoothed !== target) schedule();
  }

  function schedule() {
    if (active && !request && !document.hidden) request = requestAnimationFrame(tick);
  }

  function updateTarget() {
    target = Math.max(0, Math.min(1, scrollY / scrollRange));
    schedule();
  }

  async function loadFrames(token) {
    // Native 1284 x 716 frames come directly from the original 4-second video,
    // avoiding a downscale or another compressed-video decoding generation.
    // Pre-extraction avoids dozens of decoder seeks on the visitor's main thread.
    // On narrow screens, retain 25 frames to bound decoded image memory.
    const indices = Array.from({ length: totalFrames }, (_, i) => i)
      .filter(i => viewWidth > 767 || i % 2 === 0 || i === totalFrames - 1);
    let next = 0;
    async function worker() {
      while (next < indices.length && active && token === generation) {
        const index = indices[next++];
        try {
          const image = new Image();
          image.decoding = 'async';
          image.src = `/assets/gita-frames/frame-${String(index + 1).padStart(2, '0')}.jpg`;
          await image.decode();
          const frame = typeof createImageBitmap === 'function'
            ? await createImageBitmap(image) : image;
          if (!active || token !== generation) {
            frame.close?.();
            return;
          }
          frames.set(index, frame);
        } catch {
          // A partial/failed cache leaves the seekable MP4 as the fallback.
        }
      }
    }
    await Promise.all(Array.from({ length: 4 }, worker));
    if (!active || token !== generation) return;
    cacheReady = frames.size === indices.length;
    if (cacheReady) {
      drawn = -1;
      draw();
      schedule();
    }
  }

  function setMotionPreference() {
    const allowed = !reducedMotion.matches && !connection?.saveData;
    if (allowed === active) return;
    active = allowed;
    generation += 1;
    if (!active) {
      cancelAnimationFrame(request);
      request = 0;
      video.pause();
      video.removeAttribute('src');
      video.load();
      video.classList.remove('is-ready');
      canvas.classList.remove('is-ready');
      backdrop.classList.remove('has-frames');
      for (const frame of frames.values()) frame.close?.();
      frames.clear();
      cacheReady = false;
      drawn = -1;
      return;
    }
    measure();
    smoothed = target;
    video.src = '/assets/gita-hero.mp4';
    video.preload = 'auto';
    video.load();
    const token = generation;
    // Allow the poster and page to paint before warming the frame cache.
    setTimeout(() => {
      if (active && generation === token) loadFrames(token);
    }, 300);
    schedule();
  }

  video.addEventListener('loadeddata', () => {
    if (!active) return;
    video.classList.add('is-ready');
    schedule();
  });
  video.addEventListener('seeked', schedule);
  video.addEventListener('error', () => video.classList.remove('is-ready'));
  addEventListener('scroll', updateTarget, { passive: true });
  addEventListener('resize', measure);
  addEventListener('pageshow', updateTarget);
  document.addEventListener('visibilitychange', schedule);
  reducedMotion.addEventListener('change', setMotionPreference);
  connection?.addEventListener('change', setMotionPreference);
  new ResizeObserver(measure).observe(story);
  document.fonts.ready.then(measure);
  setMotionPreference();

  const reveals = document.querySelectorAll('[data-reveal]');
  if (!reducedMotion.matches && 'IntersectionObserver' in window) {
    document.documentElement.classList.add('has-reveal');
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-revealed');
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.15 });
    reveals.forEach((element, i) => {
      element.style.setProperty('--reveal-delay', `${Math.min(i % 3, 2) * 100}ms`);
      observer.observe(element);
    });
  }
})();
