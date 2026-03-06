"""Tests for data provider factory."""

from __future__ import annotations

import pytest

from madengine.data import get_data_provider
from madengine.data.local import LocalDataProvider
from madengine.data.nas import NASDataProvider
from madengine.data.s3 import S3DataProvider
from madengine.data.minio import MinIODataProvider


class TestGetDataProvider:
    def test_default_is_local(self):
        provider = get_data_provider({})
        assert isinstance(provider, LocalDataProvider)

    def test_local_provider(self):
        provider = get_data_provider({"provider": "local", "local": {"base_path": "/data"}})
        assert isinstance(provider, LocalDataProvider)

    def test_nas_provider(self):
        provider = get_data_provider({"provider": "nas", "nas": {"host": "nas.local"}})
        assert isinstance(provider, NASDataProvider)

    def test_s3_provider(self):
        provider = get_data_provider({"provider": "s3", "s3": {"bucket": "my-bucket"}})
        assert isinstance(provider, S3DataProvider)

    def test_minio_provider(self):
        provider = get_data_provider(
            {"provider": "minio", "minio": {"endpoint": "http://minio:9000", "bucket": "data"}}
        )
        assert isinstance(provider, MinIODataProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown data provider"):
            get_data_provider({"provider": "unknown"})
