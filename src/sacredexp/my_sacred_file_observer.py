from sacred.observers import FileStorageObserver
import pathlib, os


class CustomFileStorageObserver(FileStorageObserver):

    def save_sources(self, ex_info):
        split_src = self.source_dir.split('/')
        split_dir = self.dir.split('/')
        self.source_dir = split_src[0] + '/' + split_dir[1] + '/' + split_src[1]
        return super().save_sources(ex_info)
