import os
from pathlib import Path

from config import bucket_name, client, results_dir


def upload_to_gcs(local_file: Path):
    """Uploads a file to Google Cloud Storage"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(local_file.name)
    blob.upload_from_filename(str(local_file))
    print(f"Uploaded {local_file} to GCS")


def test_uploads():
    for experiment in range(1, 61):
        # Run your simulation (replace this with your actual code)
        print(f"Running experiment {experiment}")
        results_file = results_dir / f"result_{experiment}.txt"
        results_file.write_bytes(os.urandom(2048))  # Example file creation
        print(f"Uploading {results_file} to GCS")
        # Upload results
        upload_to_gcs(results_file)
        print(f"Uploaded {results_file} to GCS")

        # Cleanup local file
        # results_file.unlink()


if __name__ == "__main__":
    test_uploads()
