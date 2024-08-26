import jax.numpy as jnp
import numpy as np
from jax import random, device_put
import time

# Rastgele sayı üreteci anahtarını oluştur
seed = 0
key = random.PRNGKey(seed)

# Matris boyutu
size = 3000

# JAX ile rastgele bir matris oluştur (cihazda, GPU/TPU varsa otomatik olarak burada oluşturulur)
x_jnp = random.normal(key, (size, size), dtype=jnp.float32)

# NumPy ile rastgele bir matris oluştur (CPU üzerinde)
x_np = np.random.normal(size=(size, size)).astype(np.float32)

# 1) GPU'da (veya diğer hızlandırıcıda) JAX ile direkt matris çarpımı
start_time = time.time()
jnp.dot(x_jnp, x_jnp.T).block_until_ready()  # Sonuç hazır olana kadar bekle
time_1 = time.time() - start_time

# 2) CPU'da NumPy ile matris çarpımı
start_time = time.time()
np.dot(x_np, x_np.T)
time_2 = time.time() - start_time

# 3) NumPy'nin matrisini JAX ile GPU'da çarpmaya çalış (CPU'dan GPU'ya veri aktarımı gerektirir)
start_time = time.time()
jnp.dot(x_np, x_np.T).block_until_ready()  # Sonuç hazır olana kadar bekle
time_3 = time.time() - start_time

# NumPy dizisini GPU'ya açıkça aktar ve işlemi tekrar yap
x_np_device = device_put(x_np)
# 4) GPU'da JAX ile matris çarpımı (veri zaten cihazda, tıpkı 1. örnekte olduğu gibi)
start_time = time.time()
jnp.dot(x_np_device, x_np_device.T).block_until_ready()  # Sonuç hazır olana kadar bekle
time_4 = time.time() - start_time

# Sonuçları yazdır
print(f"1) GPU'da JAX ile (cihazda doğrudan): {time_1:.6f} saniye")
print(f"2) CPU'da NumPy ile: {time_2:.6f} saniye")
print(f"3) GPU'da JAX ile (NumPy dizisi, veri aktarım maliyeti): {time_3:.6f} saniye")
print(f"4) GPU'da JAX ile (cihazda, veri önceden aktarılmış): {time_4:.6f} saniye")
