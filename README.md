# QuantumComputing

Diretório destinado ao projeto da disciplina IF775 - Top Avanç. em Algoritmos (Introdução à Computação Quântica) do CIN/UFPE

**Alunos:**
- João Pedro Souza Pereira de Moura ([jpspm](mailto:jpspm@cin.ufpe.br))
- Rafael Bernardo Nunes Neto ([rbnn](mailto:rbnn@cin.ufpe.br))

perceptron, computação quântica, aprendizagem de máquina, redes neurais,
neurônio artificial, classificação, MNIST, dígitos manuscritos

Introdução
==========

Desde os anos 1940, as Redes Neurais Artificiais vêm ganhando espaço na
resolução de problemas de Inteligência Artificial. Essas redes são
formadas por várias camadas de *neurônios artificiais* - modelos
matemáticos de neurônios reais feitos especialmente para a Computação.

Em paralelo ao crescimento das Redes Neurais, surge também a Computação
Quântica, que promete acelerar processos atuais realizados por
computadores clássicos em ordem até exponencial, utilizando-se de
técnicas de Física Quântica.

Nesse contexto, nos anos 1990, emerge a junção das duas teorias: as
*Redes Neurais Quânticas*, que são construídas a partir de vários
neurônios quânticos, também conhecidos como *perceptrons*.

A partir dessa nova estrutura idealizada pela afluência dessas duas
teorias, propomos uma tarefa de classificação de dígitos manuscritos
utilizando um modelo de Perceptron Quântico, buscando mostrar o poder
que a Computação Quântica pode ter para processos de Aprendizagem de
Máquina utilizando Redes Neurais.

Ao final, é comparado o desempenho do perceptron quântico com alguns
algoritmos de redes neurais clássicas para classificação utilizados nos
dias atuais.

O Perceptron
============

Perceptron Clássico
-------------------

O primeiro perceptron clássico foi idealizado por Rosenblatt, em 1958, e
se tratava de uma camada de neurônios artificiais capazes de realizar a
tarefa de aprendizagem ao adaptar seus pesos internos dependendo da
entrada e da saída

![Componentes de um perceptron clássico](classical.jpeg){#fig:classical
width="9.5cm"}

\
desejada, com o objetivo de resolver um problema de classificação. Esse
modelo inicial foi baseado no modelo de neurônio artificial de
McCulloch-Pitts e consiste em: (1) um vetor de entrada, sobre o qual é
decodificada a entrada desejada; (2) uma camada de neurônios
artificiais, que consistem em um vetor de números reais (chamado vetor
de pesos), que são alterados de acordo com o treinamento e (3) uma
função de ativação, que recebe a combinação linear do vetor de entrada e
do vetor de pesos e classifica o resultado de acordo com um limiar
(*threshold*) determinado. O perceptron clássico é ilustrado na Figura
[1](#fig:classical){reference-type="ref" reference="fig:classical"}.

Perceptron Quântico
-------------------

A principal diferença entre o perceptron clássico e o perceptron
quântico é a forma com que a entrada é informada aos neurônios. No
perceptron clássico, as informações de input são traduzidas e fornecidas
como um vetor de números reais, os quais serão combinados linearmente
pelos neurônios e classificados de acordo com a função de ativação e o
threshold especificado.

No perceptron quântico, por sua vez, a entrada é dada por um vetor de
*qubits* (bits quânticos), que são combinações lineares da base
computacional quântica _|0>_ e _|1>_, diferente do vetor de
reais do perceptron clássico. Além disso, no perceptron quântico, para
além de utilizarmos qubits no vetor de entrada, também utilizamos essa
estrutura para os pesos dos neurônios. Assim, torna-se necessária a
aplicação de um circuito quântico para converter os inputs e pesos a
serem utilizados de valores reais em valores quânticos, utilizando-se
principalmente de portas P. Dessa forma, é possível transformar os
vetor reais de tamanho N em vetores de qubits de tamanho log<sub>2</sub>(n) que
o perceptron quântico pode entender e processar.

Metodologia
===========

Banco de Dados
--------------

O objetivo do perceptron quântico implementado foi de classificação de
dígitos manuscritos. Assim, foi utilizado o banco de dados do MNIST,
frequentemente utilizado em tarefas de processamento de imagens e na
testagem de várias ferramentas de Aprendizagem de Máquina.

A princípio, foram escolhidos os dígitos _0_ e _1_ para a classificação.
Dessa maneira, foram determinados um conjunto de treinamento, sobre o
qual são treinados os pesos e um conjunto de testes, que vai determinar
a acurácia do perceptron.

Como as imagens do MNIST dataset são originalmente formadas de
_28x28_ pixels, para que estas fossem adaptadas ao padrão de
qubits, todas as imagens foram redimensionadas para _32x32_ pixels
antes de serem alimentadas ao perceptron.

Circuito Quântico
-----------------

As imagens de dígitos manuscritos do MNIST dataset podem naturalmente
ser convertidas em arrays bidimensionais de números inteiros entre 0 e
255, de acordo com o tom de cinza daquele bit específico. Para que esses
arrays e também o array de pesos possam ser utilizados no modelo de
perceptron quântico proposto, estes devem ser transferidos para o
domínio quântico, ou seja, deve-se transformar os vetores de entrada e
de pesos em vetores de estados quânticos formados pela combinação linear
de _|0>_ e _|1>_. Para isso, é utilizado um circuito quântico.

A configuração do circuito se dá utilizando, principalmente, portas
Hadamard, portas _X_ e portas _P_ (ou *fase*), como mostrado na Figura
[2](#fig:qcircuit){reference-type="ref" reference="fig:qcircuit"} (para
o caso específico de 2 qubits de entrada). Na figura, a parte destacada
como _U<sub>i</sub>_ representa a parte responsável por introduzir os valores do
vetor de entrada, e _U<sub>w</sub>_, os pesos dos neurônios quânticos utilizados
na única camada do perceptron.

A introdução dos valores é feita principalmente pela porta _P_
controlada, aplicada somente quando todos os qubits estão

![Configuração do circuito quântico](qcircuit.png){#fig:qcircuit
width="9cm"}

\
em |1>. Assim, o circuito deve ser disposto de maneira a
introduzir os valores de pesos e de entrada apenas aos estados
correspondentes, sem afetar os outros. Isso é feito através da aplicação
de portas X sempre que o estado a ser aplicado determinado peso ou
entrada seja |1>.

Além da configuração inicial do circuito, é crucial, também, antes de
introduzir de fato os valores, normalizar o vetor de input do intervalo
_[0,255]_ para o intervalo _[0, π/2]_, para que os valores possam ser
aplicados à porta fase utilizada na inicialização.

Por fim, após a inserção do vetor de entrada e do vetor de pesos no
circuito, é feita a análise dos qubits resultantes a partir de uma porta
$X$ multicontrolada (ou porta *toffoli*). Essa porta tem o intuito de
\"transferir\" os valores dos qubits de input para um qubit auxiliar ou
*Ancilla*, ao qual é aplicado um medidor para que possam ser analisadas
as probabilidades de cada estado quântico. Assim, a partir da
distribuição de probabilidade dos estados quânticos e do threshold
determinado, é feita a classificação do resultado.

Treinamento
-----------

Para que o perceptron tenha um bom aproveitamento para dígitos não
vistos previamente, é preciso que os pesos sejam bem adaptados aos
dados. Essa etapa da Aprendizagem de Máquina é chamada de *treinamento*.
Nela, são dadas algumas imagens de exemplo para o algoritmo para que
este \"aprenda\" um suposto padrão das diferenças entre as imagens de 0
e 1 e então acerte na maioria dos casos quando receber imagens
diferentes das já vistas. Esse aprendizado é feito alterando valores do
vetor de pesos de acordo com o resultado esperado daquele input e se o
perceptron acertou ou não.

Para modificar os pesos de acordo com o resultado esperado, são
necessários um vetor inicial de pesos e também uma função de perda (ou
*loss*). Neste trabalho, foi utilizada uma instância de imagem de dígito
1 disponível no MNIST dataset como vetor de pesos inicial. Já para a
função de perda, foi utilizada a função
$$\mathscr{L}(\phi)=\frac{1}{M}\sum_{i=1}^M(y_i-\Tilde{y_i})^2$$ que
calcula a perda a cada batch de treinamento, onde $y_i$ é o resultado
esperado para aquele input, $\Tilde{y_i}$ é a saída obtida e $M$ é o
tamanho do batch.

Finalmente, dada a função de perda e o vetor de pesos inicial, foi
utilizado um otimizador SPSA (Simultaneous Perturbation Stochastic
Approximation) para otimizar o vetor de pesos de acordo com o conjunto
de treinamento elaborado.

Resultados
==========

Para avaliar os resultados, em adição à aplicação do conjunto de testes
ao perceptron quântico que foi elaborado e treinado, também foram
treinados e testados alguns de outros algoritmos clássicos usados
largamente nos dias atuais. Em seguida, foram comparados os resultado de
cada algoritmo a fim de medir o desempenho do perceptron quântico e
melhor avaliar a possibilidade da aplicação de algoritmos quânticos em
tarefas mais complexas de Aprendizagem de Máquina.

Perceptron Quântico
-------------------

Após a aplicação do otimizador ao conjunto de treinamento determinado,
tem-se o vetor de pesos final. A esperança é que, ao aplicar as imagens
do conjunto de treinamento, tenha-se um vetor de pesos tal que o
desempenho do perceptron seja o melhor possível.

Para determinar o desempenho, foi aplicado o conjunto de teste disjunto
do conjunto de treinamento e foram ocultados do perceptron os resultados
esperados. Depois de executar o algoritmo treinado para as imagens de
teste, foram comparados o vetor de resultados obtidos e o vetor dos
resultados esperados, o que, surpreendentemente, resultou numa acurácia
de 100%, um resultado bastante bom para uma rede neural de apenas 1
camada.

Perceptron Clássico
-------------------

Para avaliar o ganho de quântico para clássico, nada mais justo do que
comparar o perceptron quântico ao perceptron clássico. Assim, foi
executado um perceptron clássico da biblioteca de Python *keras* com os
mesmos conjuntos de treinamento e teste utilizados para o perceptron
quântico desenvolvido.

Após o treinamento do perceptron clássico, notou-se que este obteve um
aproveitamento de 99.3% no conjunto de testes, o que, apesar de um
resultado louvável, já demonstra algum ganho do perceptron quântico.

Redes Neurais Artificiais
-------------------------

Também foram implementadas 3 redes neurais artificiais, com o auxílio
das bibliotecas *keras*, *tensorflow* e *sklearn* de Python.

Para as 2 primeiras redes neurais treinadas, não houve mudança de
desempenho quando comparadas ao perceptron quântico. Na primeira rede
neural treinada, foram utilizadas 2 camadas de neurônios: a primeira com
128 neurônios e função de ativação relu e a segunda com 10 neurônios e
função de ativação softmax. Comparada ao perceptron clássico, essa rede
é bastante mais complexa e obteve um resultado melhor de maneira de
certo modo expectável.

Na segunda rede neural, foi utilizada a classe *MPLClassifier* da
biblioteca *sklearn*, com uma rede neural com os seguintes parâmetros:

-   hidden\_layer\_sizes=(40,),

-   max\_iter=8,

-   alpha=1e-4,

-   activation='relu',

-   solver=\"sgd\",

-   verbose=10,

-   random\_state=1,

-   learning\_rate\_init=0.2

Com isso, obteve-se um desempenho de 100% de acurácia similar ao
perceptron quântico.

Por último, e para tentar \"forçar\" um ganho positivo para o perceptron
quântico, foi criada uma última rede neural com os seguintes argumentos:

-   hidden\_layer\_sizes=(10,),

-   max\_iter=2,

-   alpha=1e-4,

-   activation='identity',

-   solver=\"adam\",

-   verbose=10,

-   random\_state=1,

-   learning\_rate='adaptive',

-   learning\_rate\_init=10

Dessa forma, com tamanho learning\_rate, houve uma perda de desempenho
dessa rede comparada ao perceptron quântico, e, ainda assim, essa rede
demonstrou uma acurácia de 96%.

Conclusão
=========

Apesar de não tratar de um problema tão abrangente, este trabalho mostra
um potencial a ser explorado no uso de redes neurais quânticas. Como foi
dito, o perceptron tem complexidade equivalente a uma rede neural de
apenas 1 camada, enquanto que vários algoritmos utilizados hoje em dia
oferecem facilmente redes neurais de múltiplas camadas e operam vários
neurônio simultaneamente. Assim, apesar da estrutura limitada do
perceptron quântico, este se mostrou comparável a notáveis algoritmos de
redes neurais usados, o que comprova o potencial de algoritmos
quânticos.

Além do ganho, podemos observar que os algoritmos clássicos, já
utilizados há muito tempo, podem ser executados em um ambiente quântico,
demonstrando que os processadores quânticos podem ser uma solução para o
eminente fim da lei de Moore.

Portanto, dada essa possibilidade de execução de algoritmos clássicos em
ambientes quânticos, os desempenhos comparáveis dos algoritmos quânticos
vistos aqui, junto dos ganhos em velocidade de execução e em uso de
memória, é notório o potencial de crescimento desses algoritmos e da
Computação Quântica no futuro.

00 Mangini, S., Tacchino, F., Gerace, D., Macchiavello, C., and Bajoni,
D. (2020). Quantum computing model of an artificial neuron with
continuously valued input data. Machine Learning: Science and
Technology, 1(4), 045008. Oxford: Clarendon, 1892, pp.68--73. Tacchino,
F., Macchiavello, C., Gerace, D. et al. An artificial neuron implemented
on an actual quantum processor. npj Quantum Inf 5, 26 (2019).
https://doi.org/10.1038/s41534-019-0140-4 Machine Learning Mastery.
(2020). Perceptron Algorithm for Classification in Python \[Online\].
Avaliable:
https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/
TensorFlow. (2022). Treine sua primeira rede neural: classificação
básica \[Online\]. Avaliable:
https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br
Scikit Learn. (2011). "Visualization of MLP weights on MNIST"
\[Online\]. Avaliable:
https://scikit-learn.org/stable/auto\_examples/neural\_networks/plot\_mnist\_filters.html\#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py
Qiskit(2022). \[Online\]. Avaliable: https://qiskit.org/
Pennylane(2022).\[Online\].Avaliable: https://pennylane.ai/
