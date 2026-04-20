import os


class TrafficStateResourceMixin:
    def _resource_exists(self, basename, suffixes):
        if basename in [None, ""]:
            return False
        if isinstance(suffixes, str):
            suffixes = (suffixes,)
        basename = str(basename)
        for suffix in suffixes:
            if os.path.exists(os.path.join(self.data_path, basename + suffix)):
                return True
        return False

    def _list_resource_basenames(self, suffixes):
        if isinstance(suffixes, str):
            suffixes = (suffixes,)
        if not os.path.exists(self.data_path):
            return []
        basenames = set()
        for filename in os.listdir(self.data_path):
            for suffix in suffixes:
                if filename.endswith(suffix):
                    basenames.add(filename[: -len(suffix)])
        return sorted(basenames)

    def _resolve_resource_basename(self, config_key, basename, suffixes, required=False):
        if basename in [None, ""]:
            basename = self.dataset
        basename = str(basename)
        if self._resource_exists(basename, suffixes):
            return basename
        fallback = str(self.dataset)
        if fallback != basename and self._resource_exists(fallback, suffixes):
            self._logger.warning(
                "Configured `%s=%s` not found under %s, fallback to `%s`.",
                config_key,
                basename,
                self.data_path,
                fallback,
            )
            return fallback
        if required:
            available = self._list_resource_basenames(suffixes)
            raise ValueError(
                "Configured `{}`={} not found under {}. Available basenames for {}: {}".format(
                    config_key, basename, self.data_path, suffixes, available
                )
            )
        return basename

    def _normalize_dataset_resource_files(self):
        data_suffixes = (".dyna", ".grid", ".od", ".gridod")
        raw_data_files = self.data_files if isinstance(self.data_files, list) else [self.data_files]
        raw_data_files = [str(item) for item in raw_data_files if item not in [None, ""]]

        # 统一要求 geo_file / rel_file / data_files 使用同一 basename
        candidate_basenames = []
        for basename in [self.dataset, self.geo_file, self.rel_file] + raw_data_files:
            if basename in [None, ""]:
                continue
            basename = str(basename)
            if basename not in candidate_basenames:
                candidate_basenames.append(basename)

        canonical_basename = None
        for basename in candidate_basenames:
            if self._resource_exists(basename, ".geo") and self._resource_exists(
                    basename, data_suffixes
            ):
                canonical_basename = basename
                break

        if canonical_basename is None:
            available_geo = self._list_resource_basenames(".geo")
            available_data = self._list_resource_basenames(data_suffixes)
            raise ValueError(
                "Cannot find a shared basename for `geo_file/rel_file/data_files` under {}. "
                "Configured: geo_file={}, rel_file={}, data_files={}. "
                "Available .geo basenames: {}. Available traffic-state basenames: {}.".format(
                    self.data_path,
                    self.geo_file,
                    self.rel_file,
                    raw_data_files,
                    available_geo,
                    available_data,
                )
            )

        if str(self.geo_file) != canonical_basename:
            self._logger.warning(
                "Aligned `geo_file` from `%s` to `%s` under %s.",
                self.geo_file,
                canonical_basename,
                self.data_path,
            )
        if str(self.rel_file) != canonical_basename:
            self._logger.warning(
                "Aligned `rel_file` from `%s` to `%s` under %s.",
                self.rel_file,
                canonical_basename,
                self.data_path,
            )
        if raw_data_files != [canonical_basename]:
            self._logger.warning(
                "Aligned `data_files` from %s to [%s] under %s.",
                raw_data_files,
                canonical_basename,
                self.data_path,
            )

        self.geo_file = canonical_basename
        self.rel_file = canonical_basename
        self.data_files = [canonical_basename]

        if self.load_external:
            self.ext_file = self._resolve_resource_basename(
                "ext_file", self.ext_file, ".ext", required=False
            )
