import sys
try:
    import google.protobuf.runtime_version
    # Disable strict gencode vs runtime version check across the test runner
    google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass
