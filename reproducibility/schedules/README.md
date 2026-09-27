# Deterministic rerun schedules

These schedules are generated with NumPy's `default_rng(seed)`. Within a
setting and seed, every compared method must consume the exact same file. The
controlled C100 participation schedules use
`make_nested_client_schedule_family`: each round starts from one shared
permutation of all 100 clients, and P10/P5/P2 use its first 10/5/2 clients.
The schedule files store each selected prefix in sorted canonical order; the
family sidecar stores a hash for every underlying round permutation. This makes
the participation comparison change the count without changing the sampling
draws.


| File | SHA-256 |
|---|---|
| c100-p10-r1000-seed20.json | 49b9df55c87f1225b768357eed26cd5d7b9f612bf596f8955dff70c4307d1dc9 |
| c100-p10-r1000-seed21.json | c3e0441487c308725fe2c1c2bd7948e3ed2cfd09a3fc29d982f19cf883172384 |
| c100-p10-r1000-seed22.json | 38301b12745ba2be54c788bce89ce65483dd6c482b6334e215bd1b01b1eaf425 |
| c100-p5-r1000-seed20.json | c82d59d03ffb85ef2a4d0b764560f237200d7d109bbd18cc490f1a76c58b8c23 |
| c100-p5-r1000-seed21.json | 6419b99f846c5cbb98e4cf55bb16adf8d8b0c4a56fc2d3caaf964c866becc18f |
| c100-p5-r1000-seed22.json | 5b5fac29d9fe00d5f1f51d3f15c2cdd38f5be340e8ada96552c5a0c6368791f5 |
| c100-p2-r1000-seed20.json | a1a909d6e34b73f914f8329f5632874a151ef14bcc187491d29507eb90a8908c |
| c100-p2-r1000-seed21.json | c5ed41dffad3c1c0467c5087d492eedf0962316a472794c1fd94e39f9b970376 |
| c100-p2-r1000-seed22.json | a1df96b0394685f4b3a0849c7301b42dae65dad5469bdf0ddb7cf3caf7450e09 |
| c500-p2-r1000-seed20.json | c5c9a3abb6dc4d03fba83b2465eed236fae239ac93eebe49536cefaa670dee09 |
| c500-p2-r1000-seed21.json | ea1ecb83d914a007603e8a7fdc5bdac67eefad40787840a09940c4bee27227ee |
| c500-p2-r1000-seed22.json | aff933995e02d518c7a7bf3501ae25a7316f879a20bf2e39907c8c4cac7ec445 |
| c100-p10-r500-seed20.json | db7efd08b72a1728de878801d940012c26506de237e9f46424927677f9d66045 |
| c100-p10-r500-seed21.json | 89e4bab2363b0c6b43c7d0bb727d5e2ea8fe8d90421686092be3e5a2a3e2db12 |
| c100-p10-r500-seed22.json | 12af5136f0cd8099f9d3349917e0c4794def2988445cf17a8d2a7ae8f1796942 |
