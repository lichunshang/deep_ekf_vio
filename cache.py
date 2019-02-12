class ImageCache(object):
    __instance = None

    def __init__(self):
        self.ImageCache = {}

    @staticmethod
    def get_instance():
        if not ImageCache.__instance:
            ImageCache.__instance = ImageCache()
        return ImageCache.__instance

    def exists(self, key):
        return key in self.ImageCache

    def get(self, key):
        return self.ImageCache[key]

    def store(self, key, val):
        self.ImageCache[key] = val


image_cache = ImageCache.get_instance()  # a global image ImageCache
