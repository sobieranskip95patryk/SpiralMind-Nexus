# 🌀 Instrukcja Instalacji - SpiralMind-Nexus

## Wymagania systemowe

### Podstawowe wymagania
- **Python 3.8+** (zalecane Python 3.10+)
- **16GB RAM** minimum (32GB zalecane dla dużych modeli)
- **CUDA 11.0+** (opcjonalne, dla akceleracji GPU)
- **10GB** wolnego miejsca na dysku

### Obsługiwane systemy operacyjne
- Windows 10/11
- macOS 10.15+ (Intel i Apple Silicon)
- Linux (Ubuntu 20.04+, CentOS 8+)

## Instalacja GPU Support (opcjonalne ale zalecane)

### NVIDIA CUDA
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# Windows - pobierz z https://developer.nvidia.com/cuda-downloads
```

### Weryfikacja CUDA
```bash
nvidia-smi
nvcc --version
```

## Krok 1: Przygotowanie środowiska

### Klonowanie repozytorium
```bash
git clone https://github.com/your-repo/SpiralMind-Nexus.git
cd SpiralMind-Nexus
```

### Utworzenie środowiska wirtualnego

#### Conda (zalecane)
```bash
# Instalacja Miniconda/Anaconda jeśli nie jest zainstalowana
# https://docs.conda.io/en/latest/miniconda.html

# Utworzenie środowiska
conda create -n spiralmind python=3.10
conda activate spiralmind

# Instalacja podstawowych pakietów
conda install numpy scipy matplotlib jupyter
```

#### venv
```bash
# Utworzenie środowiska
python -m venv venv

# Aktywacja
# Windows
venv\\Scripts\\activate
# Linux/macOS  
source venv/bin/activate
```

## Krok 2: Instalacja zależności

### Opcja 1: TensorFlow (zalecane dla beginnerów)
```bash
# CPU version
pip install tensorflow>=2.10.0

# GPU version (jeśli masz CUDA)
pip install tensorflow-gpu>=2.10.0

# Weryfikacja
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Opcja 2: PyTorch (zalecane dla zaawansowanych)
```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU version (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Weryfikacja
python -c "import torch; print(torch.cuda.is_available())"
```

### Opcja 3: JAX (dla ekspertów)
```bash
# CPU version
pip install jax jaxlib flax

# GPU version
pip install jax[cuda11_pip] flax -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Weryfikacja
python -c "import jax; print(jax.devices())"
```

### Instalacja pozostałych zależności
```bash
# Wszystkie pozostałe pakiety
pip install -r requirements.txt
```

## Krok 3: Konfiguracja projektu

### Konfiguracja środowiska
```bash
# Kopiuj przykładowe pliki konfiguracji
cp config/model_config.example.json config/model_config.json
cp config/training_config.example.json config/training_config.json

# Edytuj konfigurację
nano config/model_config.json
```

### Przykład konfiguracji (`config/model_config.json`)
```json
{
  "device": "cuda",  // "cpu", "cuda", "mps" (dla Apple Silicon)
  "precision": "float32",  // "float16", "float32", "bfloat16"
  "spiral_rnn": {
    "spiral_factor": 1.618,
    "hidden_sizes": [128, 256, 512],
    "num_layers": 3,
    "dropout": 0.1
  },
  "nexus_attention": {
    "num_heads": 8,
    "head_dim": 64,
    "spiral_scaling": true
  }
}
```

## Krok 4: Weryfikacja instalacji

### Test podstawowy
```bash
# Test importów
python -c "
import numpy as np
import tensorflow as tf  # lub torch
print('✅ Podstawowe biblioteki OK')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
"
```

### Test modułów SpiralMind
```bash
# Test struktury projektu
python -c "
import sys
sys.path.append('src')
try:
    from models.spiral_rnn import SpiralRNN
    print('✅ Moduły SpiralMind OK')
except ImportError as e:
    print(f'❌ Błąd importu: {e}')
"
```

### Test kompletny
```bash
# Uruchom test integracyjny
python tests/integration_test.py

# Test z przykładowymi danymi
python examples/quickstart.py
```

## Krok 5: Pierwsze uruchomienie

### Przykład Hello World
```python
# examples/hello_spiral.py
import sys
sys.path.append('src')

from models.spiral_rnn import SpiralRNN
import numpy as np

# Utworzenie prostego modelu spiralnego
model = SpiralRNN(
    input_dim=10,
    hidden_dim=32,
    spiral_layers=2
)

# Wygenerowanie przykładowych danych
sample_input = np.random.randn(1, 5, 10)  # batch, sequence, features

# Forward pass
output = model(sample_input)
print(f"✅ Model spiralny działa! Output shape: {output.shape}")
```

```bash
python examples/hello_spiral.py
```

### Demo z wizualizacją
```bash
# Uruchom demo spiralnych wzorców
python examples/spiral_visualization.py

# Powinno otworzyć okno z wizualizacją spirali
```

## Rozwiązywanie problemów

### Problem: CUDA Out of Memory
```bash
# Zmniejsz batch size w konfiguracji
# Lub użyj gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Problem: Slow Training
```bash
# Włącz mixed precision training
# W config/training_config.json:
{
  "mixed_precision": true,
  "compile_model": true  // dla PyTorch 2.0+
}
```

### Problem: Import Errors
```bash
# Reinstaluj w odpowiedniej kolejności
pip uninstall -y numpy scipy scikit-learn
pip install numpy==1.24.0
pip install scipy scikit-learn
pip install -r requirements.txt
```

### Problem: Apple Silicon (M1/M2)
```bash
# Użyj Metal Performance Shaders
pip install tensorflow-metal

# Lub PyTorch z MPS
pip install torch torchvision torchaudio

# Test MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Optymalizacja wydajności

### Dla TensorFlow
```python
# Włącz XLA compilation
import tensorflow as tf
tf.config.optimizer.set_jit(True)

# Mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Dla PyTorch
```python
# Compilation (PyTorch 2.0+)
model = torch.compile(model)

# Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Monitoring zasobów
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Pamięć Python
pip install memory-profiler
python -m memory_profiler examples/memory_test.py

# CPU profiling
pip install py-spy
py-spy top --pid $(pgrep python)
```

## Następne kroki

1. **Przeczytaj dokumentację**: `docs/ARCHITEKTURA.md`
2. **Zobacz przykłady**: `examples/`
3. **Uruchom benchmarki**: `python tests/benchmark.py`
4. **Dołącz do społeczności**: Discord SpiralMind Community

## Wsparcie techniczne

- **GitHub Issues**: [Zgłoszenia błędów](https://github.com/your-repo/SpiralMind-Nexus/issues)
- **Dokumentacja**: [docs/](docs/)
- **Discord**: [SpiralMind Community](https://discord.gg/spiralmind)
- **Email**: support@spiralmind-nexus.ai

---

✅ **Gratulacje! Twój system SpiralMind-Nexus jest gotowy do eksploracji spiralnych architektur AI!**