include MODEL.mk

# ************************************ VENV ************************************ #
BASE := /raid/speng65
WANDB_ACCOUNT := anthonypeng

SHELL := /bin/bash
VENV_NAME := robustarch
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate $(VENV_NAME)
PYTHON := $(CONDA_ACTIVATE) && python
PIP := $(CONDA_ACTIVATE) && pip

# Install advertorch from github due to its usage of obsoleted code from pytorch
# https://github.com/BorealisAI/advertorch/issues/99
clean:
	rm -rf lib/*
	rm -f .venv_done

.venv_done: clean
	conda create -n $(VENV_NAME) python=3.9
	$(PIP) install -r requirements.txt
	cd lib && git clone https://github.com/NVIDIA/apex
	$(PIP) install -v --disable-pip-version-check --no-cache-dir lib/apex
	cd lib && git clone https://github.com/BorealisAI/advertorch.git
	$(PIP) install -e lib/advertorch
	$(PIP) install -e .
	touch $@

# ************************ Adversarial Training Configs ************************ #
SRC = robustarch
assign-vars = $(foreach A,$2,$(eval $1: $A))

# FAT - 3 Phases
PHASE1 = cd $(SRC) && $(PYTHON) -m main train_test=fat_phase1 ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet-sz/160"
PHASE2 = cd $(SRC) && $(PYTHON) -m main train_test=fat_phase2 ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet-sz/352"
PHASE3 = cd $(SRC) && $(PYTHON) -m main train_test=fat_phase3 ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet"
TEST = cd $(SRC) && $(PYTHON) -m main train_test=test ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT)
TRAIN_EPS4 = ++attack.train.eps=5 ++attack.test.eps=4

# Standard PGD AT
TRAIN_PGD = cd $(SRC) && $(PYTHON) -m main train_test=at_pgd attack=train ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet"
TEST_PGD = cd $(SRC) && $(PYTHON) -m main train_test=at_pgd_test attack=test ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet"
TRAIN_PGD3_EPS4 = ++attack.train.eps=4 ++attack.train.gamma=2.67 ++attack.test.eps=4 ++attack.train.step=3
STEP_LR1 = ++train_test.schedule="step" ++train_test.decay_t=30 ++train_test.decay_rate=0.1 ++train_test.warmup_epochs=10 ++train_test.warmup_lr=0.1 ++train_test.lr=0.1
COLOR_JITTER = ++train_test.color_jitter=0.1
LIGHTING = ++train_test.lighting=true
TRAIN_CONFIG1 = $(TRAIN_PGD3_EPS4) $(STEP_LR1) $(COLOR_JITTER) $(LIGHTING) ++attack.train.eps_schedule=null ++train_test.cooldown_epochs=0

# *********************************** Attacks ********************************** #
# PGD
TEST_PGD10_2-1 = ++attack.test.eps=2 ++attack.test.gamma=1 ++attack.test.step=10
TEST_PGD10_4-1 = ++attack.test.eps=4 ++attack.test.gamma=1 ++attack.test.step=10
TEST_PGD10_8-1 = ++attack.test.eps=8 ++attack.test.gamma=2 ++attack.test.step=10
TEST_PGD50_4-1 = ++attack.test.eps=4 ++attack.test.gamma=1 ++attack.test.step=50
TEST_PGD100_2-1 = ++attack.test.eps=2 ++attack.test.gamma=1 ++attack.test.step=100
TEST_PGD100_4-1 = ++attack.test.eps=4 ++attack.test.gamma=1 ++attack.test.step=100
TEST_PGD100_8-1 = ++attack.test.eps=8 ++attack.test.gamma=1 ++attack.test.step=100

# AutoAttack
AUTOATTACK_4 = ++attack.test.name=aa ++attack.test.batch_size=256 ++attack.test.n_examples=5000 ++attack.test.eps=4

# ******************************** EXPERIMENTS ********************************* #
# ************************************ FAT ************************************* #
# ResNet-50
NAME = Torch_ResNet50
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=Torch_ResNet50 ARCH=$(RESNET50_TORCH))
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TORCHMODEL) $(TRAIN_EPS4)
	$(PHASE2) $(TORCHMODEL) $(TRAIN_EPS4)
	$(PHASE3) $(TORCHMODEL) $(TRAIN_EPS4)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=Torch_ResNet50 ARCH=$(RESNET50_TORCH))
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_2-1)
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_4-1)
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_8-1)
	touch $@

# WideResNet50-2
NAME = Torch_Wide_ResNet50_2
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=Torch_Wide_ResNet50_2 ARCH=$(WIDE_RESNET50_2_TORCH))
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TORCHMODEL) $(TRAIN_EPS4)
	$(PHASE2) $(TORCHMODEL) $(TRAIN_EPS4)
	$(PHASE3) $(TORCHMODEL) $(TRAIN_EPS4)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=Torch_Wide_ResNet50_2 ARCH=$(WIDE_RESNET50_2_TORCH))
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_2-1)
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_4-1)
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_8-1)
	touch $@

# RobArch-S
NAME = RobArch_S
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=RobArch_S)
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_S)
	$(PHASE2) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_S)
	$(PHASE3) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_S)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=RobArch_S)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_2-1) $(RESNET50) $(ROBARCH_S)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(ROBARCH_S)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_8-1) $(RESNET50) $(ROBARCH_S)
	touch $@

# RobArch-M
NAME = RobArch_M
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=RobArch_M)
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_M)
	$(PHASE2) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_M)
	$(PHASE3) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_M)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=RobArch_M)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_2-1) $(RESNET50) $(ROBARCH_M)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(ROBARCH_M)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_8-1) $(RESNET50) $(ROBARCH_M)
	touch $@

# RobArch-L
NAME = RobArch_L
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=RobArch_L)
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_L) ++train_test.lr_values.1=0.25
	$(PHASE2) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_L)
	$(PHASE3) $(TRAIN_EPS4) $(RESNET50) $(ROBARCH_L)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=RobArch_L)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_2-1) $(RESNET50) $(ROBARCH_L)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(ROBARCH_L)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_8-1) $(RESNET50) $(ROBARCH_L)
	touch $@

# ****************************** Standard PGD AT ******************************* #
# RobArch-S
NAME = PGDAT_RobArch_S
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=PGDAT_RobArch_S)
experiments/$(NAME)/.done_train:
	$(TRAIN_PGD) $(TRAIN_CONFIG1) $(RESNET50) $(ROBARCH_S)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=PGDAT_RobArch_S)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(ROBARCH_S)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD50_4-1) $(RESNET50) $(ROBARCH_S)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_4-1) $(RESNET50) $(ROBARCH_S)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_8-1) $(RESNET50) $(ROBARCH_S)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_2-1) $(RESNET50) $(ROBARCH_S)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_aa, NAME=PGDAT_RobArch_S)
experiments/$(NAME)/.done_test_aa: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_4) $(RESNET50) $(ROBARCH_S)
	touch $@

ROBARCH_S_WEIGHTS = trained_models/pretrained/robarch_s.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=PGDAT_RobArch_S)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(ROBARCH_S_WEIGHTS))", "")
	$(info RobArch-S weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/RobArch/resolve/main/pretrained/robarch_s.pt -P trained_models/pretrained
	$(info RobArch-S weights file are successfully downloaded.)
endif
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_4) $(RESNET50) $(ROBARCH_S) ++train_test.resume="../$(ROBARCH_S_WEIGHTS)"
	touch $@

# RobArch-M
NAME = PGDAT_RobArch_M
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=PGDAT_RobArch_M)
experiments/$(NAME)/.done_train:
	$(TRAIN_PGD) $(TRAIN_CONFIG1) $(RESNET50) $(ROBARCH_M)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=PGDAT_RobArch_M)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_8-1) $(RESNET50) $(ROBARCH_M)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_2-1) $(RESNET50) $(ROBARCH_M)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(ROBARCH_M)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD50_4-1) $(RESNET50) $(ROBARCH_M)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_4-1) $(RESNET50) $(ROBARCH_M)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_aa, NAME=PGDAT_RobArch_M)
experiments/$(NAME)/.done_test_aa: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_4) $(RESNET50) $(ROBARCH_M)
	touch $@

ROBARCH_M_WEIGHTS = trained_models/pretrained/robarch_m.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=PGDAT_RobArch_M)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(ROBARCH_M_WEIGHTS))", "")
	$(info RobArch-M weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/RobArch/resolve/main/pretrained/robarch_m.pt -P trained_models/pretrained
	$(info RobArch-M weights file are successfully downloaded.)
endif
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_4) $(RESNET50) $(ROBARCH_M) ++train_test.resume="../$(ROBARCH_M_WEIGHTS)"
	touch $@

# RobArch-L
NAME = PGDAT_RobArch_L
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=PGDAT_RobArch_L)
experiments/$(NAME)/.done_train:
	$(TRAIN_PGD) $(TRAIN_CONFIG1) $(RESNET50) $(ROBARCH_L)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=PGDAT_RobArch_L)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(ROBARCH_L)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD50_4-1) $(RESNET50) $(ROBARCH_L)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_4-1) $(RESNET50) $(ROBARCH_L)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_8-1) $(RESNET50) $(ROBARCH_L)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_2-1) $(RESNET50) $(ROBARCH_L)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_aa, NAME=PGDAT_RobArch_L)
experiments/$(NAME)/.done_test_aa: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_4) $(RESNET50) $(ROBARCH_L)
	touch $@

ROBARCH_L_WEIGHTS = trained_models/pretrained/robarch_l.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=PGDAT_RobArch_L)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(ROBARCH_L_WEIGHTS))", "")
	$(info RobArch-L weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/RobArch/resolve/main/pretrained/robarch_l.pt -P trained_models/pretrained
	$(info RobArch-L weights file are successfully downloaded.)
endif
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_4) $(RESNET50) $(ROBARCH_L) ++train_test.resume="../$(ROBARCH_L_WEIGHTS)"
	touch $@

# ****************************************************************************** #