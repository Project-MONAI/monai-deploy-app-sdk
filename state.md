# Driver Upgrade Handoff

## Host and Goal

- Host OS: Ubuntu 22.04.5 LTS (Jammy), not RHEL.
- Hardware: 8x NVIDIA A100-SXM4-40GB (`sm_80`) in an NVSwitch DGX system.
- Secure Boot: disabled.
- Current driver: 570.211.01 (CUDA 12.8 capability).
- CUDA toolkit installed: 13.2.
- Goal: upgrade the development host to NVIDIA R610 so CUDA 13 and Holoscan workloads can run locally before H100 deployment.

## Why Upgrade

The Python environment contains `torch 2.13.0+cu130`. Under driver 570.211.01 it reports a driver-too-old warning and `torch.cuda.is_available()` is `False`.

## Package Investigation

- NVIDIA CUDA Ubuntu repository is configured.
- From driver branch 590 onward, NVIDIA removed branch suffixes from its package names.
- Do not use `nvidia-driver-610`; that is an older Ubuntu package and does not contain NVIDIA's R610.57.04 package.
- Use unversioned `nvidia-driver` and `nvidia-fabricmanager` pinned to the same version.
- The successful dry run was:

```bash
sudo apt-get -s install \
  nvidia-driver=610.57.04-1ubuntu1 \
  nvidia-fabricmanager=610.57.04-1ubuntu1
```

It removes the legacy R550/R570 `-server` packages and installs a consistent R610.57.04 NVIDIA stack. It retains `cuda-toolkit-13-2`. It also installs normal desktop-related dependencies because `nvidia-driver` is the complete NVIDIA meta-package.

## Upgrade Procedure

1. Drain/stop GPU jobs. GPUs 4-7 previously ran vLLM workers, which will be terminated by reboot.
2. Optional but recommended: install NVIDIA's exact-version pin:

```bash
sudo apt-get install nvidia-driver-pinning-610.57.04
```

3. Install the driver and matched Fabric Manager:

```bash
sudo apt-get install \
  nvidia-driver=610.57.04-1ubuntu1 \
  nvidia-fabricmanager=610.57.04-1ubuntu1
```

4. Reboot:

```bash
sudo reboot
```

Do not use the NVIDIA `.run` installer and do not run `apt autoremove` as part of this upgrade.

## Post-Reboot Validation

```bash
nvidia-smi
systemctl status nvidia-fabricmanager --no-pager
/tmp/monai-env/.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: `nvidia-smi` shows driver 610.57.04; Fabric Manager is active; PyTorch reports CUDA available and identifies an A100.

## Failure Recovery

If `nvidia-smi` cannot communicate with the driver, check the DKMS build and boot the previous kernel from GRUB if necessary. The installed R570 package state was captured in `~/nvidia-packages-before-upgrade.txt` if that snapshot command was run.
