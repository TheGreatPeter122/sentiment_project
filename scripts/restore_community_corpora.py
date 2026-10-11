"""Verify and restore the bundled public corpora without network or upstream code."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def restore(root, destination, verify_only=False):
    root = root.resolve()
    destination = destination.resolve()
    package = root / 'data/community_corpora'
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    checked = 0
    for archive in manifest['archives']:
        archive_path = (root / archive['path']).resolve()
        if not archive_path.is_relative_to(root):
            raise ValueError(f'Archive path escapes project: {archive["path"]}')
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != archive['sha256']:
            raise ValueError(f'Archive checksum mismatch: {archive_path}')
        target_root = (destination / archive['restore_to']).resolve()
        if not target_root.is_relative_to(destination):
            raise ValueError(f'Restore directory escapes destination: {archive["restore_to"]}')
        files = {entry['path']: entry for entry in archive['files']}
        with zipfile.ZipFile(archive_path) as bundle:
            if len(bundle.infolist()) != len(files) or set(bundle.namelist()) != set(files):
                raise ValueError(f'Duplicate, missing or unexpected entries in {archive_path}')
            for member in bundle.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or '..' in name.parts or '\\' in member.filename or ':' in member.filename:
                    raise ValueError(f'Unsafe archive path: {member.filename}')
                target = (target_root / Path(*name.parts)).resolve()
                if not target.is_relative_to(target_root.resolve()):
                    raise ValueError(f'Archive path escapes destination: {member.filename}')
                data = bundle.read(member)
                entry = files[member.filename]
                if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError(f'Member checksum mismatch: {member.filename}')
                if not verify_only:
                    if target.exists():
                        if target.read_bytes() != data:
                            raise FileExistsError(f'Preserving an existing modified file: {target}')
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                checked += 1
        print(f'Verified {archive_path.name}: {len(files)} files')
    if not verify_only:
        destination_manifest = (destination / 'data/community_corpora/manifest.json').resolve()
        if not destination_manifest.is_relative_to(destination):
            raise ValueError('Manifest path escapes destination')
        if destination_manifest.resolve() != (package / 'manifest.json').resolve():
            if destination_manifest.exists() and destination_manifest.read_bytes() != (package / 'manifest.json').read_bytes():
                raise FileExistsError(f'Preserving an existing manifest: {destination_manifest}')
            destination_manifest.parent.mkdir(parents=True, exist_ok=True)
            destination_manifest.write_bytes((package / 'manifest.json').read_bytes())
    print(f'{checked} file checks passed; ' + ('no extraction requested' if verify_only else f'restored to {destination}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check every ZIP and member SHA256 without extracting.')
    parser.add_argument('--destination', type=Path, help='Alternate project root for a clean restore.')
    arguments = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    restore(project, (arguments.destination or project).resolve(), arguments.verify_only)
