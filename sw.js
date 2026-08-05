const CACHE = 'sweden-mushrooms-v4';
const APP_URLS = [
  './',
  'index.html',
  'manifest.json',
  'assets/style.css',
  'assets/icon-192.png',
  'assets/icon-512.png',
];
const RASTER_URLS = [
  'assets/habitat-sweden/forest-mask.png',
  'assets/habitat-sweden/forest-coverage.png',
  'assets/habitat-sweden/forest-state.png',
  'assets/habitat-sweden/forest-reference.png',
];
for (const species of ['cib', 'tr', 'black', 'regalis', 'matsutake']) {
  RASTER_URLS.push(`assets/habitat-sweden/${species}-score.png`);
  for (const threshold of [40, 60, 75]) {
    RASTER_URLS.push(`assets/habitat-sweden/${species}-overlay-${threshold}.png`);
  }
  RASTER_URLS.push(
    `assets/habitat/${species}-score.png`,
    `assets/habitat/${species}-overlay.png`,
  );
}
RASTER_URLS.push(
  'assets/habitat/forest-mask.png',
  'assets/habitat/soil-category.png',
  'assets/habitat/soil-overlay.png',
);

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll([...APP_URLS, ...RASTER_URLS])));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key))))
      .then(() => clients.claim()),
  );
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.pathname.includes('/assets/habitat')) {
    event.respondWith(
      caches.match(event.request).then(cached => cached || fetch(event.request).then(response => {
        if (response.ok) caches.open(CACHE).then(cache => cache.put(event.request, response.clone()));
        return response;
      })),
    );
    return;
  }

  const cacheable = url.origin === self.location.origin ||
    ['unpkg.com', 'cdn.jsdelivr.net'].includes(url.hostname);
  event.respondWith(
    fetch(event.request).then(response => {
      if (cacheable && response.ok) {
        caches.open(CACHE).then(cache => cache.put(event.request, response.clone()));
      }
      return response;
    }).catch(async () => {
      const cached = await caches.match(event.request);
      if (cached) return cached;
      if (event.request.mode === 'navigate') return caches.match('index.html');
      throw new Error(`Offline and not cached: ${event.request.url}`);
    }),
  );
});
