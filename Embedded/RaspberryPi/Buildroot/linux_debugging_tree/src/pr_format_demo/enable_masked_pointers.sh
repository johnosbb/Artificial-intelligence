#!/bin/sh
echo 2 | sudo tee /proc/sys/kernel/kptr_restrict # Always mask pointers

