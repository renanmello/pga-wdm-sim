# Paralelização do Algoritmo de RWA em Redes WDM

Este projeto paraleliza um algoritmo de roteamento e alocação de comprimento de onda (RWA) em redes WDM utilizando `mpi4py`. A solução foi desenvolvida no contexto de um Trabalho de Conclusão de Curso (TCC) e visa otimizar o tempo de execução do código original. O projeto faz uso de bibliotecas fundamentais como `NetworkX`, `NumPy`, `mpi4py` e `Matplotlib`.

## Índice

- [Introdução](#introdução)
- [Instalação](#instalação)
- [Execução](#execução)
- [Referências](#referências)

## Introdução

O roteamento e alocação de comprimentos de onda (RWA) é um problema central em redes de comunicação óptica, como redes WDM (Wavelength Division Multiplexing). Através da paralelização com MPI (Message Passing Interface), é possível melhorar a eficiência do algoritmo, distribuindo as tarefas entre vários processadores e reduzindo o tempo de execução para redes maiores.

Este projeto é baseado no código original disponível [neste repositório](https://github.com/cassiotbatista/rwa-wdm-sim), que foi modificado para aproveitar a execução paralela.

## Instalação

Para rodar este projeto, é necessário ter Python instalado, bem como as bibliotecas abaixo:

- `networkx`
- `numpy`
- `mpi4py`
- `matplotlib`

### Passos para instalação:

1. Clone o repositório:

git clone https://github.com/renanmello/pga-wdm-sim.git
cd seu-repositorio

2. Crie um ambiente virtual (opcional, mas recomendado):

python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows

3. Instale as dependências:

pip install -r requirements.txt

Instalação do MPI:

Para executar o código paralelizado, também é necessário instalar uma implementação do MPI, como o MPICH ou o OpenMPI. Para instalá-lo:

Em distribuições Linux (Debian/Ubuntu):

sudo apt-get install mpich

Em sistemas macOS usando Homebrew:

    brew install mpich

Em sistemas Windows, siga este guia.

Execução

Após a instalação do MPI e das dependências do projeto, o código pode ser executado com múltiplos processos utilizando o comando mpiexec.
Exemplo de execução:

mpiexec -n 4 python main.py

Neste exemplo, o código será executado utilizando 4 processos paralelos. Ajuste o número de processos (-n) de acordo com o ambiente disponível.
Referências

Este projeto é baseado no código original disponível em rwa-wdm-sim, desenvolvido por Cassio Batista. A versão atual foi modificada para incluir paralelização com MPI usando a biblioteca mpi4py.
