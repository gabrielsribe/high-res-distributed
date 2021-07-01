## Gabriel de Souza Ribeiro
## g234836@dac.unicamp.br
## RA: 234836

# Trabalho MO833

## Introdução
Em telecomunicações, infraestrutura de tecnologia da informação e na engenharia de software, escalabilidade é uma característica desejável em todo o sistema, rede ou processo, que indica a capacidade de manipular uma porção crescente de trabalho de forma uniforme, ou estar preparado para crescer (ANDRE B. Bondi, 2000). Desta forma, com o advento do paradigma da nuvem computacional se faz necessário que aplicações desenvolvidas possuam a capacidade de aumentar sua capacidade de processamento para que estejam preparadas para se adaptar a cargas de trabalho dos mais variados tamanhos, visando por exemplo atingir um nível de otimização de custos desejável ao contexto da aplicação. Sendo assim, o objetivo desse trabalho é implementar um sistema de otimização de dinâmica de recursos computacionais utilizando a API do CLAP. O sistema e sua implementação deverá permitir iniciar e terminar nós computacionais dinamicamente baseados na eficiência das Paramount Iterations durante a execução de uma aplicação de aprendizado de máquina.

## Aplicação Escolhida
A aplicação escolhida para o trabalho foi uma Rede Adversária Generativa para gerar imagens de Super Resolução a partir de imagens de menor resolução.
Uma Rede Adversária Generativa ou, GAN (Generative adversarial Network) é uma classe de estruturas de aprendizagem de máquinas projetada por Ian Goodfellow e seus colegas em 2014 (Goodfellow et al, 2014) , onde as duas redes disputam entre si um jogo onde uma tenta “enganar” a outra, para que os resultados da primeira sejam cada vez melhores.
Para isso foi selecionado um código pré-implementado do artigo “Esrgan: Enhanced super-resolution generative adversarial networks” (Wang, Xintal et al, 2018) , o código pode ser encontrado
aqui: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/esrgan.py

A aplicação conta com uma série de parametros personalizaveis, sendo eles:
* epoch, type=int, default=0, help="epoch to start training from"
* n_epochs, type=int, default=2, help="number of epochs of training"
* dataset_name, type=str, default="img_align_celeba", help="name of the dataset"
* batch_size, type=int, default=1, help="size of the batches"
* lr, type=float, default=0.0002, help="adam: learning rate"
* b1, type=float, default=0.9, help="adam: decay of first order momentum of gradient"
* b2, type=float, default=0.999, help="adam: decay of first order momentum of gradient"
* decay_epoch, type=int, default=10, help="epoch from which to start lr decay"
* n_cpu, type=int, default=8, help="number of cpu threads to use during batch generation"
* hr_height, type=int, default=128, help="high res. image height"
* hr_width, type=int, default=128, help="high res. image width"
* channels, type=int, default=3, help="number of image channels"
* sample_interval, type=int, default=5, help="interval between saving image samples"
* checkpoint_interval, type=int, default=5000, help="batch interval between model checkpoints"
* residual_blocks, type=int, default=23, help="number of residual blocks in the generator"
* warmup_batches, type=int, default=500, help="number of batches with pixel-wise loss only"
* lambda_adv, type=float, default=5e-3, help="adversarial loss weight"
* lambda_pixel, type=float, default=1e-2, help="pixel-wise loss weight"
* local_rank, type=int, help="Local rank. Necessary for using the torch.distributed.launch utility."
* *cuda, action='store_true', help='enables cuda'

## Adaptações necessárias e ferramental desenvolvido

O código pré-implementado não possuía a capacidade de ser rodado de maneira distribuída, para isso foi necessário que fossem feitas adaptações para que a aplicação fosse inicializada utilizando o torchdistributed.
Também foi desenvolvido um container docker para que a aplicação pudesse ser inicializada com o ferramental necessário e garantir sua portabilidade. 
A fim de coletar os dados para o experimento o código também foi adaptado para coletar tempos de execução das épocas, iterações e sua inicialização.
Sumarizando seguem as alterações que foram realizadas/desenvolvidas
* (Aplicação) Configuração da aplicação para rodar distribuidamente (torchdistributed)
* (Aplicação) Configuraçao da aplicação para catalogar e exportar os tempos de execução (Inicialização, Iterations e Epocas)
* (Docker) Imagem Docker para rodar a aplicação
* (AWS) Configuração de security group (gsribeiro-2-sg)
* (AWS) Configuração da imagem padrão (AMI image)
* (Clap) Configuraçao de roles, instances etc.
* (Experimento) Jupyter notebook com os experimentos

## Experimentos propostos

Inicialmente o objetivo do experimento era avaliar a performance do otimizador dos recursos computacionais, comparado ao tempo ótimo, ou seja rodando a aplicação com o conjunto de máquinas considerado ideal para aplicação. Para isso o experimento inciaria com um cluster formado de máquinas heterogeas e deveria terminar com um conjunto de máquinas homogeneas consideradas ideais para a aplicação. As métricas que seriam avaliadas seriam as métricas de tempo de iterations, epocas, tempo de inicialização e tempo de setup das máquinas para a otimização do cluster e seleção de máquinas ideais.


Entretanto, conforme o desenvolvimento e testes da aplicação na AWS, notou-se que o torch distributed roda paralelamente em máquinas diferentes, porém sincronamente, ou seja os tempos das iterations e epocas são bem parecidos porque o processamento roda sincronamente nos nós. Foi discutido em aula que, uma estratégia válida seria encontrar uma configuração de máquina ideal e propor ao usuário sua utilização.

Sendo assim, como eu estava também com receio de estourar os créditos da minha conta aws (Na atividade anterior do clap acabei esquecendo de terminar umas máquinas) acabei por adotar a estratégia de rodar pequenos batchs de dados em máquinas com configurações diferentes, mantendo a mesma massa de dados.

Após coletar os dados das execuções realizei a estimativa média para se como a aplicação tivesse rodado por uma hora, depois avaliando essa relação de tempo de run da instancia dado 1 dolar (paramont iterations por dólar)


Para validar o experimento, será coletado os tempos de inicialização e execução da iterações e épocas, utilizado o dataset CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) que provê um conjunto de imagens de celebridades agrupadas em atributos faciais como, por exemplo, presença de óculos e ondulação de cabelos.

Os cenários de teste foram pensados de acordo com as limitações da com AWS disponibilizada na disciplina, então, não serão realizados cenários onde o tempo de processamento ou a volumetria de imagens sejam muito grandes.

Sendo assim temos duas massas de dados onde é possivel utilizar no treinamento da rede, sendo:
* data/data_experiment = 500 imagens
* data/img_align_celeba_test = 100 imagens

### Cenários de teste
#### Teste 1
##### Avaliar a performance da máquina c4x
* Rodar a massa de dados img_align_celeba_test utilizando os nucleos disponiveis
* Coletar o resultado
* Analisar o custo de paramont iterations por dolar

#### Teste 2
##### Avaliar a performance da máquina c5x

#### Massa de dados
* Rodar a massa de dados img_align_celeba_test utilizando os nucleos disponiveis
* Coletar o resultado
* Analisar o custo de paramont iterations por dolar

## Resultados e Discussão

Os nomes dos arquivos de resultados (*.out) tem em sua composição (TEST_NODE-TEST_EXPERIMENT_ID) pois a abordagem inicial seria utilizar esses valores para fazer o o fetch de resultados doo cluster.

Como nos experimentos não foi utilizado cluster, esses campos não foram preenchidos.

As informações detalhadas da run e como rodar o experimento podem ser encontradas no notebook experiment_execute.ipynb

A conta foi feita da seguinte maneira:
* node_count = numero de nós de processamento (Cores)
* duração_média_de_cada_iteration = (epoch_time) / (iteration_count * node_count * 1.0)
* segundos_disponiveis_na_instancia_com_um_dolar = seconds_with_one_dolar = ((1.0 / preço_por_hora_da_instancia) * 3600) * node_count
* numero_de_iteraçoes_media_por_dolar = (segundos_disponiveis_na_instancia_com_um_dolar - tempo_de_inicializaçao) / (duração_média_de_cada_iteration)

Note que o tempo de inicialização foi tratado como uma constante que foi retirada do tempo total disponivel em uma instancia com 1 dolar

Foi constastado que a execução mais proveitosa seria na máquina c5x com uma média de 55507 iterations por dólar, enquanto na máquina c4x foi obtido uma média de 12162 iterations por dólar.

Foi constatado também que, neste código, ele utiliza de um modelo pré-treinado ou "calibrado", não entendo muito bem o conceito por trás, mas no inicio da aplicação ele faz o download desse arquivo na máquina hospedeira. Talvez fosse necessário considerar isso nas contas dos preços das máquinas, mas dado o deadline do trabaho e sabendo que era um valor constante essa análise não foi realizada. Mas é importante ao leitor/usuário saber que existe esse processo no inicio, caso for instanciar um cluster com inumeras máquinas 

Não foi possivel avaliar a perfomance ou custo do optmizador, mas neste cenário esse número seria quanto foi gasto para realizar o teste em cada máquina, sendo assim:
* Custo da run na instancia c4x = tempo_total_do_experimento (58.6192) em dólar = 0,003239278
* Custo da run na instancia c5x = tempo_total_do_experimento (59.6940) em dólar = 0,002814444
* Custo total da "optimização" = 0,006053722 dólar

# Aos senhores docentes da Disciplina MO833A
Primeiramente, muito obrigado por toda a paciência e ajuda em todo esse semestre ao professor Edson Borin e também ao monitor Otávio pela disponibilidade impar.

Saimos dos conceitos do que era processamento distribuido, depois nuvem computacional, containers, algoritmos de aprendizado de máquina até conseguirmos rodar esse experimento. Foi muito aprenzidado (E dores nas costas) em muito pouco tempo neste semestre.

Espero também que tenha consiguido atingir as espectativas dos senhores com relação a minha entrega.

O meu muito obrigado.

Abraços!


