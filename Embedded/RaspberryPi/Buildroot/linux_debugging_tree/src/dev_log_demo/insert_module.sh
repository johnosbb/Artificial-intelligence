#!/bin/bash
sudo insmod dev_log_demo.ko
echo "Module inserted. Use 'dmesg' or 'journalctl -k' to view logs."
