pip install gdown
gdown https://drive.google.com/uc?id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2
gdown https://drive.google.com/uc?id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3
gdown https://drive.google.com/uc?id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_
gdown https://drive.google.com/uc?id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG
gdown https://drive.google.com/uc?id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO
gdown https://drive.google.com/uc?id=17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP
gdown https://drive.google.com/uc?id=1XoaGG3ek26YLFvGzmkKeOz54INW0fruR
gdown https://drive.google.com/uc?id=16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg
mkdir data
mv camelyonpatch_level_2_split_* data
cd data
gzip -d camelyonpatch_level_2_split_*
mkdir pcam
mv camelyonpatch_level_2_split_* pcam
cd ..
