
##### Build Docker
docker build -t ml-sha/nudge_docker .

##### Run Docker
docker run --gpus all -it -v /home/ml-jsha/storage:/NUDGE/storage --rm ml-sha/nudge_docker

#### Train Atari 2600 Games

``` 
python -m atari_py.import_roms /NUDGE/storage/ROMS

```

``` 
python -m pi.train_dqn_objctive -m Pong --teacher_game_nums 1000 --device 6
python -m pi.train -m Kangaroo --train_epochs 100000 --resume --device 6
python -m pi.train -m Asterix --train_epochs 50000 --resume --device 7
python -m pi.train -m Pong --train_epochs 100000 --episode_num 1000 --resume --device 7
python -m pi.train_mlp_hla -m Pong --train_epochs 20000 --episode_num 1000 --resume --device 7
python -m pi.play -m Pong --train_epochs 20000 --episode_num 1000 --resume --device 7
python -m pi.expanding -m Freeway --device 8 --start_frame 5000 --end_frame 10000
python -m pi.expanding -m Freeway --device 9 --start_frame 10000 --end_frame 15000
python -m pi.expanding -m Freeway --device 10 --start_frame 15000 --end_frame 25000
python -m pi.expanding -m Freeway --device 11 --start_frame 25000 --end_frame 35000

python -m nesy_pi.aaa_main -m Freeway --device 10 --with_pi
python -m nesy_pi.aaa_main -m Freeway --device 2 --with_pi --show_process
python -m nesy_pi.aaa_main -m Asterix --device 1 --with_pi --show_process

python -m nesy_pi.aaa_main -m getout --device 1 --with_pi --show_process

python -m nesy_pi.play_with_clauses -m Freeway --device 11 
python -m nesy_pi.collect_data -m Freeway --device 2
python -m nesy_pi.collect_data_getout -m getout --device 10
python -m nesy_pi.collect_asterix -m Asterix --device 0

python -m nesy_pi.train_nudge -m getout -alg logic -env getout -r getout_pi --with_pi --device 1 -s 2
python -m nesy_pi.train_nudge -m atari -alg logic -env Freeway -r freeway_pi --with_pi --device 2 -s 1
python -m nesy_pi.train_nudge -m atari -alg logic -env Asterix -r asterix_pi --with_pi --device 1 -s 0
```