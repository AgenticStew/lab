$ErrorActionPreference = "Stop"
$R = "C:\spec-test"
New-Item -ItemType Directory -Force "$R\models","$R\llama","$R\dl" | Out-Null
# llama.cpp b12189, Windows CUDA 13.5 (same build as the 5090 tests)
foreach ($z in "llama-b12189-bin-win-cuda-13.5-x64.zip","cudart-llama-bin-win-cuda-13.5-x64.zip") {
  if (-not (Test-Path "$R\dl\$z")) { curl.exe -sSL -o "$R\dl\$z" "https://github.com/ggml-org/llama.cpp/releases/download/b12189/$z" }
  Expand-Archive -Force "$R\dl\$z" "$R\llama" }
# model files at the pinned revisions (model_sources.json), downloaded in parallel
Start-Process -NoNewWindow -FilePath curl.exe -ArgumentList "-sSL","-C","-","-o","$R\models\Qwen3.8-27B-UD-Q4_K_M.gguf","https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf" -PassThru | Out-Null
Start-Process -NoNewWindow -FilePath curl.exe -ArgumentList "-sSL","-C","-","-o","$R\models\mtp-Qwen3.8-27B-Q4_0.gguf","https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4af51b3542f775b0a8e4b310bfb3f17e01c306c9/MTP/mtp-Qwen3.8-27B-Q4_0.gguf" -PassThru | Out-Null
Start-Process -NoNewWindow -FilePath curl.exe -ArgumentList "-sSL","-C","-","-o","$R\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf","https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/resolve/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf" -PassThru | Out-Null
Start-Process -NoNewWindow -FilePath curl.exe -ArgumentList "-sSL","-C","-","-o","$R\models\Qwen3.5-9B-Q4_K_M.gguf","https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf" -PassThru | Out-Null
Start-Process -NoNewWindow -FilePath curl.exe -ArgumentList "-sSL","-C","-","-o","$R\models\Llama-3.2-1B-Instruct-Q8_0.gguf","https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q8_0.gguf" -PassThru | Out-Null
while (Get-Process curl -ErrorAction SilentlyContinue) { Start-Sleep 5 }
$want = @{
  "Qwen3.8-27B-UD-Q4_K_M.gguf" = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"
  "mtp-Qwen3.8-27B-Q4_0.gguf" = "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e"
  "Qwen3.8-27B-DFlash2-Q4_K_M.gguf" = "1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd"
  "Qwen3.5-9B-Q4_K_M.gguf" = "e8dd94817e95d6c0939102049d068418269978377b13616c4726235e232841fe"
  "Llama-3.2-1B-Instruct-Q8_0.gguf" = "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"
}
foreach ($n in $want.Keys) { $h = (Get-FileHash -Algorithm SHA256 "$R\models\$n").Hash.ToLower(); "{0} {1}" -f ($(if ($h -eq $want[$n]) {"OK "} else {"BAD"}), $n) }
cmd /c "$R\llama\llama-server.exe --version 2>&1" | Select-Object -First 3
"SETUP_DONE"
