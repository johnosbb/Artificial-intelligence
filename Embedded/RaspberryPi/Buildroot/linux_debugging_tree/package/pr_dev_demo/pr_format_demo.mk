################################################################################
# pr_format_demo.mk - Buildroot package for pr_format_demo module
################################################################################

PR_FORMAT_DEMO_VERSION = 1.0
PR_FORMAT_DEMO_SITE = $(BR2_EXTERNAL_LINUX_DEBUGGING_TREE_PATH)/src/format_specifiers
PR_FORMAT_DEMO_SITE_METHOD = local
PR_FORMAT_DEMO_DEPENDENCIES = linux

define PR_FORMAT_DEMO_BUILD_CMDS
    $(MAKE) ARCH=$(KERNEL_ARCH) CROSS_COMPILE=$(TARGET_CROSS) -C $(LINUX_DIR) M=$(@D) modules
endef

define PR_FORMAT_DEMO_INSTALL_TARGET_CMDS
    $(MAKE) ARCH=$(KERNEL_ARCH) CROSS_COMPILE=$(TARGET_CROSS) -C $(LINUX_DIR) \
        M=$(@D) INSTALL_MOD_PATH=$(TARGET_DIR) modules_install
endef


define PR_FORMAT_DEMO_LINUX_CONFIG_FIXUPS
	$(call KCONFIG_ENABLE_OPT,CONFIG_MODULE_UNLOAD)
endef

$(eval $(kernel-module))
$(eval $(generic-package))
