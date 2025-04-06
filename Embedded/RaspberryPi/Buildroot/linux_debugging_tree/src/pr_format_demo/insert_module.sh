#!/bin/sh
dmesg | grep pr_format_demo

echo "Module inserted. Use 'dmesg' or 'journalctl -k' to view logs."
