🚀 README.md
🛡️ Cybersecurity LLM Lab (LLaMA GGUF) - Conda Environment
This repository sets up a clean, local environment for using LLaMA GGUF models (e.g., Llama-3.2-1B-Instruct-Q4_K_M-GGUF) for offline, fast, privacy-preserving LLM workflows on Windows/Linux using Conda.

📌 Features
✅ Isolated Conda environment for stability.
✅ Supports llama-cpp-python (GPU/CPU) for easy scripting.
✅ Ready for LangChain + RAG workflows for cybersecurity data.
✅ RTX 4060 / Ryzen 7 tested (adjustable for other GPUs/CPUs).
✅ Can be extended with fine-tuning + quantization pipelines.

🛠️ Folder Structure
plaintext
Copy
Edit
cyber_ai_llama/
│
├── models/
│   └── Llama-3.2-1B-Instruct-Q4_K_M.gguf
│
├── scripts/
│   ├── test_inference.py
│   └── rag_pipeline.py          # optional
│
├── data/                        # if using RAG
├── results/                     # logs, outputs
└── README.md
⚙️ Prerequisites
✅ Miniconda or Anaconda installed.
✅ git and cmake if you plan to build llama.cpp CLI.

🚩 Step 1: Create and Activate Conda Environment
bash
Copy
Edit
conda create -n cyber_llm python=3.11
conda activate cyber_llm
🚩 Step 2: Install Dependencies
For CPU-only:

bash
Copy
Edit
pip install llama-cpp-python
For GPU (CUDA):

bash
Copy
Edit
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
Optional (if using RAG workflows):

bash
Copy
Edit
pip install langchain chromadb
🚩 Step 3: Download the Model
Download your GGUF model (e.g., Llama-3.2-1B-Instruct-Q4_K_M.gguf) from Hugging Face and place it in:

Copy
Edit
models/
🚩 Step 4: Test Inference
Create scripts/test_inference.py:

python
Copy
Edit
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=8,
    n_gpu_layers=-1  # Use -1 for GPU, 0 for CPU
)

response = llm("Q: What is cybersecurity?\nA:", max_tokens=200)
print(response["choices"][0]["text"])
Run:

bash
Copy
Edit
python scripts/test_inference.py
🚩 Step 5: Managing Your Environment
Save the environment:
bash
Copy
Edit
conda list --explicit > cyber_llm_env.txt
Recreate environment on another machine:
bash
Copy
Edit
conda create --name cyber_llm --file cyber_llm_env.txt
🚩 Optional: Build llama.cpp CLI
For direct CLI usage:

bash
Copy
Edit
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
Run:

bash
Copy
Edit
./main -m ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Explain CVE-2023-12345." -n 200
✅ Troubleshooting
If you get:

go
Copy
Edit
error loading model: create_tensor: tensor 'output.weight' not found
Check your .gguf file integrity using:

powershell
Copy
Edit
Get-FileHash .\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf -Algorithm SHA256
and re-download if corrupted.

If GPU not utilized, ensure:

Correct CMAKE_ARGS on install.

nvidia-smi detects your GPU.

Latest CUDA toolkit + drivers are installed.

✨ Next Goals
✅ Integrate with LangChain + Chroma for a local cybersecurity RAG assistant.
✅ Test on larger GGUF models (7B/8B) if VRAM allows.
✅ Fine-tune LLaMA on cybersecurity datasets, re-quantize, and load for private testing.

🚀 Ready to experiment:
You now have a clean, structured environment to build your Cybersecurity LLM Assistant locally, fully private, and offline.

If you would like, I can generate:
✅ A test_inference.py
✅ A rag_pipeline.py for learning local RAG with your LLaMA GGUF model.

for your next learning steps. Let me know!
