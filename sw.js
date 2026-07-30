const CACHE = 'sweden-mushrooms-v1';
const RASTER_URLS = [
  'assets/habitat-sweden/forest-mask.png',
  'assets/habitat-sweden/forest-coverage.png',
  'assets/habitat-sweden/forest-state.png',
  'assets/habitat-sweden/forest-reference.png',
];
for (const s of ['cib','tr','black','regalis','matsutake']) {
  RASTER_URLS.push(`assets/habitat-sweden/${s}-score.png`);
  for (const t of [40,60,75]) RASTER_URLS.push(`assets/habitat-sweden/${s}-overlay-${t}.png`);
}
for (const s of ['cib','tr','black','regalis','matsutake']) {
  RASTER_URLS.push(`assets/habitat/${s}-score.png`, `assets/habitat/${s}-overlay.png`);
}
RASTER_URLS.push(
  'assets/habitat/forest-mask.png',
  'assets/habitat/soil-category.png',
  'assets/habitat/soil-overlay.png'
);

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(RASTER_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(clients.claim());
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Cache-first for raster assets (works on both root and subpath hosting)
  if (url.pathname.includes('/assets/habitat')) {
    e.respondWith(
      caches.match(e.request).then(cached => cached || fetch(e.request).then(res => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, copy));
        }
        return res;
      }))
    );
    return;
  }
  // Network-first for everything else
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});