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

## Adaptações necessárias e ferramental desenvolvido

O código pré-implementado não possuía a capacidade de ser rodado de maneira distribuída e/ou escalável, para isso foi necessário que fossem feitas adaptações para que a aplicação fosse inicializada utilizando o torchdistributed.
Também foi desenvolvido um container docker para que a aplicação pudesse ser inicializada com o ferramental necessário e garantir sua portabilidade. 
A fim de coletar os dados para o experimento o código também foi adaptado para coletar tempos de execução das épocas, iterações e sua inicialização

## Experimentos propostos

O objetivo do experimento é avaliar a performance do otimizador dos recursos computacionais, comparado ao tempo ótimo, ou seja rodando a aplicação com o conjunto de máquinas considerado ideal para aplicação. Para isso o experimento inciará com um cluster formado de máquinas heterogeas e deverá terminar com um conjunto de máquinas homogeneas consideradas ideais para a aplicação. Serão avaliadas as métricas de tempo de iterations, epocas, tempo de inicialização e tempo de setup das máquinas para a otimização do cluster e seleção de máquinas ideais.
Para validar o experimento, será coletado os tempos de inicialização e execução da iterações e épocas, utilizado o dataset CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) que provê um conjunto de imagens de celebridades agrupadas em atributos faciais como, por exemplo, presença de óculos e ondulação de cabelos.

Os cenários de teste foram pensados de acordo com as limitações da com AWS disponibilizada na disciplina, então, não serão realizados cenários onde o tempo de processamento ou a volumetria de imagens sejam muito grandes.

### Cenários de teste
#### Teste 1
##### Avaliar a homogenização do cluster
Cluster inicial 3 máquinas heretogeneas: 

#### Teste 2
Avaliar o tempo de e custo rodando nas máquinas ideais
##### Cluster inicial 3 máquinas homogeneas: [Serão descobertas no experimento]

#### Massa de dados
1000 Imagens do dataset CelebA de 128x128