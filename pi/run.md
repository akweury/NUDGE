
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

python -m pi.train_dqn_objctive -m Asterix --teacher_game_nums 1000 --device 8
python -m pi.play -m Asterix --episode_num 1000 --train_epochs 100000 --device 12
python -m pi.play --teacher_agent pretrained -m Boxing --with_explain --device 10 --teacher_game_nums 100
```