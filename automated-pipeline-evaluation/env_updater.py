#!/usr/bin/env python3
"""
Environment File Updater

Safely updates .env file with pipeline-specific configuration
while preserving all other environment variables.
"""

import os
import sys
from typing import Dict
import shutil
from datetime import datetime


class EnvUpdater:
    """Handles safe updates to .env files"""

    def __init__(self, env_path: str = ".env"):
        self.env_path = env_path
        self.backup_path = f"{env_path}.backup"

    def update(self, updates: Dict[str, str], create_backup: bool = True) -> bool:
        """
        Update .env file with new values.

        Args:
            updates: Dictionary of {KEY: value} to update
            create_backup: Whether to create backup before updating

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup if requested
            if create_backup and os.path.exists(self.env_path):
                shutil.copy2(self.env_path, self.backup_path)

            # Read current .env file
            if os.path.exists(self.env_path):
                with open(self.env_path, 'r') as f:
                    lines = f.readlines()
            else:
                lines = []

            # Track which keys we've updated
            updated_keys = set()
            new_lines = []

            # Update existing lines
            for line in lines:
                stripped = line.strip()

                # Skip empty lines and comments
                if not stripped or stripped.startswith('#'):
                    new_lines.append(line)
                    continue

                # Check if this line is one we need to update
                if '=' in line:
                    key = line.split('=')[0].strip()
                    if key in updates:
                        new_lines.append(f"{key}={updates[key]}\n")
                        updated_keys.add(key)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            # Add any new keys that weren't in the file
            for key, value in updates.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={value}\n")

            # Write updated file
            with open(self.env_path, 'w') as f:
                f.writelines(new_lines)

            return True

        except Exception as e:
            print(f"Error updating .env: {e}", file=sys.stderr)
            # Restore from backup if it exists
            if create_backup and os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.env_path)
            return False

    def restore_backup(self) -> bool:
        """Restore .env from backup"""
        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.env_path)
                return True
            return False
        except Exception as e:
            print(f"Error restoring backup: {e}", file=sys.stderr)
            return False

    def get_current_value(self, key: str) -> str:
        """Get current value of a key from .env"""
        try:
            with open(self.env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        if '=' in line:
                            current_key, value = line.split('=', 1)
                            if current_key.strip() == key:
                                return value.strip()
        except Exception:
            pass
        return ""


def main():
    """CLI interface for env updater"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Update .env file")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--updates", type=str, help="JSON string of updates")
    parser.add_argument("--key", type=str, help="Single key to update")
    parser.add_argument("--value", type=str, help="Value for single key")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup")
    parser.add_argument("--get", type=str, help="Get current value of key")

    args = parser.parse_args()

    updater = EnvUpdater(args.env_file)

    # Get mode
    if args.get:
        value = updater.get_current_value(args.get)
        print(value)
        return

    # Update mode
    updates = {}

    if args.updates:
        # JSON mode: multiple updates
        updates = json.loads(args.updates)
    elif args.key and args.value:
        # Single key-value mode
        updates[args.key] = args.value
    else:
        print("Error: Must provide --updates JSON or --key/--value pair", file=sys.stderr)
        sys.exit(1)

    # Perform update
    success = updater.update(updates, create_backup=not args.no_backup)

    if success:
        print(f"Successfully updated {len(updates)} variables in {args.env_file}")
        for key, value in updates.items():
            print(f"  {key}={value[:50]}{'...' if len(value) > 50 else ''}")
    else:
        print(f"Failed to update {args.env_file}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
