################################################################################
# log_demo.mk - Buildroot package for log_demo module
################################################################################

LOG_DEMO_VERSION = 1.0
LOG_DEMO_SITE = $(BR2_EXTERNAL_LINUX_DEBUGGING_TREE_PATH)/src/log_demo
LOG_DEMO_SITE_METHOD = local
LOG_DEMO_DEPENDENCIES = linux

define LOG_DEMO_BUILD_CMDS
    $(MAKE) ARCH=$(KERNEL_ARCH) CROSS_COMPILE=$(TARGET_CROSS) -C $(LINUX_DIR) M=$(@D) modules
endef

define LOG_DEMO_INSTALL_TARGET_CMDS
    $(MAKE) ARCH=$(KERNEL_ARCH) CROSS_COMPILE=$(TARGET_CROSS) -C $(LINUX_DIR) \
        M=$(@D) INSTALL_MOD_PATH=$(TARGET_DIR) modules_install
endef


define LOG_DEMO_LINUX_CONFIG_FIXUPS
	$(call KCONFIG_ENABLE_OPT,CONFIG_MODULE_UNLOAD)
endef

$(eval $(kernel-module))
$(eval $(generic-package))
