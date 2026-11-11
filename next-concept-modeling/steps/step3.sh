#!/usr/bin/env bash
# TESTPLAN step 3, inside the container, network on. torch stays the image's 2.10.0+cu128; no flash-attn.
# Deviation (h): no gh / git clone; the code comes from the pinned tarballs in /dl/src, checked against
# SHA256SUMS (fetched read-only on the host at the pinned SHAs, never run there), and #5 from its saved diff.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.get_arch_list())" | tee /results/env.txt
python -c "
import sys, torch
bad = []
if sys.version_info < (3, 11): bad.append('Python ' + sys.version.split()[0] + ' < 3.11')
if torch.__version__ != '2.10.0+cu128': bad.append('torch ' + torch.__version__ + ', not 2.10.0+cu128')
if 'sm_120' not in torch.cuda.get_arch_list(): bad.append('sm_120 missing from the arch list')
sys.exit('STOP: ' + '; '.join(bad)) if bad else print('env OK')
"
(cd /dl/src && sha256sum -c SHA256SUMS) | tee /results/src-sha256-check.txt
mkdir -p /work
rm -rf /work/ncm /work/duo  # a rerun starts from the pristine tarballs (and re-applies #5 once)
tar -xzf /dl/src/ncm-c702742.tar.gz -C /work
mv /work/ncm-c70274273f489abc73832b2b98584a8483b6b203 /work/ncm
tar -xzf /dl/src/duo-7c9b498.tar.gz -C /work
mv /work/duo-7c9b498f5b717de064d6fad7e2509c866e6cb620 /work/duo
cat /work/ncm/requirements.txt /work/duo/requirements.txt
# deviation (m): also drop torchaudio, torchvision and triton (DUO pins torch-2.3-era builds of them)
grep -viE '^(torch|torchaudio|torchvision|flash.?attn|transformers|triton)([=<> ]|$)' /work/duo/requirements.txt > /work/duo-reqs-nodeps.txt
cat /work/duo-reqs-nodeps.txt
pip install transformers==5.22.2
pip install --no-deps -r /work/ncm/requirements.txt
pip install --no-deps -r /work/duo-reqs-nodeps.txt
# deviation (j): --no-deps leaves the pins' own dependencies out; install only what pip check names, torch held
echo "torch==2.10.0" > /work/constraints.txt
pip check | tee /results/pip-check-before.txt || true
missing=$(grep -oE 'requires [A-Za-z0-9_.-]+' /results/pip-check-before.txt | awk '{print $2}' | sort -u | grep -viE '^(torch|torchaudio|torchvision|triton|flash.?attn|nvidia-.*)$' || true)
if [ -n "$missing" ]; then
  echo "installing what pip check named: $missing"
  pip install -c /work/constraints.txt $missing
fi
if ! python -c "import lightning" 2>/results/lightning-import.txt; then
  echo "lightning 2.2.1 does not import on torch 2.10; trying 2.5.1"
  pip install -c /work/constraints.txt --no-deps lightning==2.5.1
  python -c "import lightning"
fi
# hydra-core 1.3.2 and omegaconf 2.3.0 need the ANTLR 4.9 runtime (the image ships 4.13.2, which can't read their grammar)
pip install -c /work/constraints.txt antlr4-python3-runtime==4.9.3
python -c "from omegaconf import OmegaConf; c = OmegaConf.create({'a': 1, 'b': '\${a}'}); assert c.b == 1; print('omegaconf interpolation OK')"
pip check | tee /results/pip-check-after.txt || true
python -c "import torch; assert torch.__version__ == '2.10.0+cu128', torch.__version__; print('torch held', torch.__version__)"
pip freeze | tee /results/pip-freeze.txt
# #5 (B04): the tarball has no git history to search, so that is recorded, then the reviewed diff is applied
echo "c702742 tarball: no git history; #5 is not in c702742 (GitHub, see ncm-pr5.json); applying its saved diff" | tee /results/ncm-pr5.txt
cp /dl/src/ncm-pr5.json /results/ncm-pr5.json
cp /dl/src/ncm-pr5.patch /work/ncm-pr5.patch
sha256sum /work/ncm-pr5.patch | tee -a /results/ncm-pr5.txt
cd /work/ncm
# the image has neither git nor patch: bench/apply_patch.py applies the diff strictly (exact hunks, no fuzz)
python -I /bench/apply_patch.py /work/ncm-pr5.patch /work/ncm --check
python -I /bench/apply_patch.py /work/ncm-pr5.patch /work/ncm | tee -a /results/ncm-pr5.txt
echo "#5 cached-path lines in models/next_concept.py: $(grep -c '_kv_prefix\|_kv_level' models/next_concept.py)" | tee -a /results/ncm-pr5.txt
