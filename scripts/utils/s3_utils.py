import os
from typing import Optional
import dotenv

try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # boto3 not installed -> S3 silently disabled
    boto3 = None
    ClientError = Exception  # type: ignore


dotenv.load_dotenv()

_S3_CLIENT = None


def s3_enabled() -> bool:
    """
    S3 is considered enabled if boto3 is available and WEATHER_S3_BUCKET is set.
    """
    return boto3 is not None and bool(os.getenv("WEATHER_S3_BUCKET"))


def _client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        if boto3 is None:
            raise RuntimeError("boto3 is not installed (required for S3 mode)")
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _bucket_and_prefix() -> tuple[str, str]:
    bucket = os.getenv("WEATHER_S3_BUCKET")
    if not bucket:
        raise RuntimeError("WEATHER_S3_BUCKET is not set")
    prefix = os.getenv("WEATHER_S3_PREFIX", "weather-plus").strip("/")
    return bucket, prefix


def _join_key(*parts: str) -> str:
    clean = [str(p).strip("/") for p in parts if p is not None and str(p).strip("/")]
    return "/".join(clean)


def upload_file(
    local_path: str, subdir: str, key: Optional[str] = None
) -> Optional[str]:
    """
    Upload local_path to s3://WEATHER_S3_BUCKET/WEATHER_S3_PREFIX/<subdir>/<filename>
    Returns the object key, or None if S3 disabled.
    """
    if not s3_enabled():
        return None

    bucket, base = _bucket_and_prefix()
    if key is None:
        name = os.path.basename(local_path)
        key = _join_key(base, subdir, name)

    cli = _client()
    cli.upload_file(local_path, bucket, key)
    return key


def upload_bytes(
    data: bytes,
    subdir: str,
    name: str,
    content_type: Optional[str] = None,
    extra_args: Optional[dict] = None,
) -> Optional[str]:
    """
    Upload bytes directly as an S3 object under WEATHER_S3_PREFIX/<subdir>/<name>.
    Returns the object key, or None if S3 disabled.
    """
    if not s3_enabled():
        return None

    bucket, base = _bucket_and_prefix()
    key = _join_key(base, subdir, name)
    args = extra_args.copy() if extra_args else {}
    if content_type:
        args["ContentType"] = content_type
    _client().put_object(Bucket=bucket, Key=key, Body=data, **args)
    return key


def object_exists(subdir: str, name: str) -> bool:
    """Return True if WEATHER_S3_PREFIX/<subdir>/<name> exists on S3."""
    if not s3_enabled():
        return False
    bucket, base = _bucket_and_prefix()
    key = _join_key(base, subdir, name)
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def download_if_exists(local_path: str, subdir: str, key: Optional[str] = None) -> bool:
    """
    If an object for local_path exists on S3, download it and return True.
    Otherwise return False (no error).
    """
    if not s3_enabled():
        return False

    bucket, base = _bucket_and_prefix()
    if key is None:
        name = os.path.basename(local_path)
        key = _join_key(base, subdir, name)

    cli = _client()
    try:
        cli.head_object(Bucket=bucket, Key=key)
    except ClientError as e:  # not found -> False
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    cli.download_file(bucket, key, local_path)
    return True


def download_subdir_to(
    local_dir: str,
    subdir: str,
    prefix_filter: str = "",
    suffix: str = "",
) -> None:
    """
    Download all objects under WEATHER_S3_PREFIX/<subdir>/ into local_dir,
    optionally filtered by filename prefix/suffix.
    """
    if not s3_enabled():
        return

    bucket, base = _bucket_and_prefix()
    cli = _client()
    s3_prefix = _join_key(base, subdir) + "/"

    os.makedirs(local_dir, exist_ok=True)

    paginator = cli.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = os.path.basename(key)
            if prefix_filter and not name.startswith(prefix_filter):
                continue
            if suffix and not name.endswith(suffix):
                continue
            dest = os.path.join(local_dir, name)
            if not os.path.exists(dest):
                cli.download_file(bucket, key, dest)


def download_json(subdir: str, name: str) -> Optional[dict]:
    """
    Download a JSON file from S3 and return it as a dict.
    Returns None if the file doesn't exist or S3 is disabled.
    """
    if not s3_enabled():
        return None
    
    import json
    bucket, base = _bucket_and_prefix()
    key = _join_key(base, subdir, name)
    cli = _client()
    try:
        resp = cli.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def list_objects(subdir: str, prefix: str = "", suffix: str = "") -> list[str]:
    """
    List object keys under WEATHER_S3_PREFIX/<subdir>/ matching prefix/suffix.
    Returns list of full S3 URIs (s3://bucket/key).
    """
    if not s3_enabled():
        return []
    
    bucket, base = _bucket_and_prefix()
    cli = _client()
    s3_prefix = _join_key(base, subdir) + "/"
    
    uris = []
    paginator = cli.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = os.path.basename(key)
            if prefix and not name.startswith(prefix):
                continue
            if suffix and not name.endswith(suffix):
                continue
            uris.append(f"s3://{bucket}/{key}")
    return uris


def list_object_keys(subdir: str, prefix: str = "", suffix: str = "") -> list[str]:
    """
    List object keys (relative to WEATHER_S3_PREFIX/<subdir>/) matching prefix/suffix.
    Returns list of relative keys (not full S3 URIs).
    """
    if not s3_enabled():
        return []
    
    bucket, base = _bucket_and_prefix()
    cli = _client()
    s3_prefix = _join_key(base, subdir) + "/"
    
    keys = []
    paginator = cli.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Remove the base prefix to get relative key
            if key.startswith(base + "/"):
                rel_key = key[len(base) + 1:]
            else:
                rel_key = key
            name = os.path.basename(key)
            if prefix and not name.startswith(prefix):
                continue
            if suffix and not name.endswith(suffix):
                continue
            keys.append(rel_key)
    return keys