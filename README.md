# stark-nist-fips
ESORICS Code supporting paper - FIPS-Aligned STARKs with Concrete Multi-Level Post-Quantum Security

Results 

Benchmarks on AWS c5.4xlarge
(Cascade Lake, AVX-512, 16 Rayon threads)

Visualisation of benchmarks:  
https://saholmes.github.io/stark-nist-fips

Code

Code is written in RUST and uses the Rayon threads capability to accelerate parallel operations.  
Arkworks scaffolding is used and we have created our own Golidlocks code within the Arkworks framework.  
Note:  We use the posiedon algebraic hash for acceleration from the Arkworks library.  

Rerunning the benchmarks:  

from within the Cargo/channel directory 

In your environment set the number of Rayon threads to the capability of you computer

export RAYON_NUM_THREADS=16

or

set RAYON_NUM_THREADS=16

Within the directory
crates/channel/benches

The file:
multi_air_end_to_end.rs

Edit the file to select field extension and r value for NIST level (as per paper)

This sets the field extension
Default is FP^6

On line 30:  

// Choose between Sextic and Optic

use deep_ali::sextic_ext::SexticExt;

type Ext = SexticExt;

//use deep_ali::octic_ext::OcticExt;

//type Ext = OcticExt;

For the r value: Edit the value at line 162 to choose one value 
From Paper, NIST L1 Security
let r: usize = 54;

From Paper, NIST L3 Security
let r: usize = 79;

From Paper, NIST L5 Security
let r: usize = 105;
Using SHA3-256 

Note from paper, for Level 1 
q= 2^40 adversary, 
use SHA3-256
q = 2^65 adversary
use SHA3-384
q = 2^90
use SHA3-512

Note from paper, for Level 1 
q= 2^40 adversary, 
use SHA3-384
q = 2^65 adversary
use SHA3-384
q = 2^90
use SHA3-512

Note from paper, for Level 5 
q= 2^40 adversary, 
use SHA3-512
q = 2^65 adversary
use SHA3-512
q = 2^90
Not achievable 


Running code:  

Using SHA3-256

cargo clean

cargo bench \
  --features parallel,sha3-256 \
  --bench multi_air_end_to_end \
  -- --sample-size 20 --measurement-time 10

Using SHA3-384

cargo clean

cargo bench \
  --features parallel,sha3-384 \
  --bench multi_air_end_to_end \
  -- --sample-size 20 --measurement-time 10  

Using SHA3-512

cargo clean

cargo bench \
  --features parallel,sha3-512 \
  --bench multi_air_end_to_end \
  -- --sample-size 20 --measurement-time 10  


When completed, results per run will be in one common file:

benchmarkdata.csv

You will then need to edit this with the specific environment you used for the run for example AVX512, Mac Mini etc.  



