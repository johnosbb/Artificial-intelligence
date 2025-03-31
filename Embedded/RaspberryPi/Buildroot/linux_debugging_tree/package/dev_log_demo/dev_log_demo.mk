################################################################################
# dev_log_demo.mk - Buildroot package for dev_log_demo module
################################################################################

DEV_LOG_DEMO_VERSION = 1.0
DEV_LOG_DEMO_SITE = $(BR2_EXTERNAL_LINUX_DEBUGGING_TREE_PATH)/src/dev_log_demo
DEV_LOG_DEMO_SITE_METHOD = local


define DEV_LOG_DEMO_BUILD_CMDS
    $(MAKE) ARCH=$(KERNEL_ARCH) CROSS_COMPILE=$(TARGET_CROSS) -C $(LINUX_DIR) M=$(@D) modules
endef

define DEV_LOG_DEMO_INSTALL_TARGET_CMDS
    $(MAKE) ARCH=$(KERNEL_ARCH) CROSS_COMPILE=$(TARGET_CROSS) -C $(LINUX_DIR) \
        M=$(@D) INSTALL_MOD_PATH=$(TARGET_DIR) modules_install
endef


define DEV_LOG_DEMO_LINUX_CONFIG_FIXUPS
	$(call KCONFIG_ENABLE_OPT,CONFIG_MODULE_UNLOAD)
endef

$(eval $(kernel-module))
$(eval $(generic-package))
