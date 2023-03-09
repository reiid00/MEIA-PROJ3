
# Divisão de Tarefas
Seguindo o diagrama, [Vista_Fisica](https://github.com/reiid00/MEIA-PROJ3/blob/main/Documentation/VistaFisica.drawio.png), teremos inicialmente duas redes neuronais cujo output será o input de uma terceira e última rede neuronal (que ajusta o texto do guião e devolve a resposta ao ticket). 
Assim, sugiro dividirmos o grupo em 2 partes, uma parte para  desenvolver a componente de tipificação dos tickets e outra para desenvolver a componente de análise de sentimento/emoção de texto.
Ambas estas redes necessitarão de dois datasets distintos.

### NN Tipificação
Relativamente à rede de tipificação, esta deverá incluir um guião/manual de suporte e deverá ser treinada com um dataset com diversas categorias, sendo que já encontrámos um bastante interessante, penso que o Hugo já enviou para todos em cc, menos para a Stefane (falta-nos saber o email do isep). 
Este dataset é composto por múltiplas colunas e, tendo em conta o contexto do nosso projeto, inclui 4 categorias relevantes, nomeadamente "product", "subproduct", "issue" e "subissue", o que indica que à partida teremos 4 nós de saída nesta rede neuronal. 
Além disso, o dataset também inclui o texto enviado pelo cliente/ticket e a resposta fornecida por uma dada empresa, referenciada noutra coluna, sendo que existem tickets de múltiplas empresas.

#### Passos a Seguir
O primeiro passo será gerar o guião a partir das respostas genéricas, dado que o dataset parece possuir respostas iguais ou semelhantes para tickets associados a valores iguais nas tais 4 categorias. 
Isto é, em princípio, o dataset segue um guião, mas estenão é fornecido diretamente no dataset, só que nós deveremos conseguir obtê-lo indiretamente, a partir desta assunção.
Posteriormente, teremos que treinar uma rede neuronal com os tais 4 nós de saída, capaz de prever então as 4 categorias, possibilitando a obtenção da resposta a partir do guião, gerado no passo anterior.
Por fim, enviamos essa informação, isto é, a resposta do guião, para a última rede neuronal, que é a última fase do nosso projeto.


### NN Sentiment Analysis
Relativamente à rede de análise de sentimento, esta tem como objetivo prever o sentimento/estado emocional da pessoa que gerou um determinado texto através da análise do conteúdo do mesmo.
Esta rede deve ser treinada de forma a puder ser utilizada para qualquer contexto de suporte, isto é, deverá ser geral, não específica, ao contrário da outra que depende do dataset e guião.
Assim, deverá prever o estado emocional do texto gerado de tickets de um produto ou serviço qualquer, independentemente da sua área/categoria, isto é, deve ser capaz de generalizar. 
É importante salientar que esta fase irá envolver um estudo mais aprofundado do estado da arte, dado que envolve teoria e conceitos bem fundamentados/explorado, sendo que é importante que todo o processo de R&D seja bem documentado no artigo. 
Por fim, o output desta rede será também fornecido à camada final, cujo desenvolvimento será iniciado após pelo menos uma das redes estar completa.


## Divisão da Equipa
NN Tipificação - Filipe; Hugo

Sentiment Analysis - Vasco; Isadora; Stefane


### Blog
Como a maioria já sabe (todos menos a Stefane), todos os grupos têm de escrever resumos semanais relativos ao seu progresso, mas, desta vez, sugiro uma abordagem diferente. 

#### 1ª e 2ª Semanas
Relativamente às duas primeiras semanas, eu posso escrevê-las, sendo que se traduzem na fase de brainstorming, confirmação de ideias entre equipa e com professores,
e, também, incluindo a procura de datasets.

#### 3ª a 8ª Semanas
Relativamente às restantes semanas, já incluindo esta, 3ª, sugiro divirmos todos os reports semanais em 2 componentes, uma focada exclusivamente no progresso da análise de sentimento e outra no progresso da tipificação e criação do guião. 
Nas últimas semanas, onde à partida, se tudo correr bem, estaremos todos a trabalhar em conjunto na componente final que gere o texto de resposta, se calhar fará mais sentido decidirmos nessa altura como dividir as respetivas semanas.


### Briefing Semanal
Considero importante reunirmos todas as semanas, pelo menos 30m, com o objetivo de ficarmos a par do progresso de ambas as componentes, sendo que, também, claro, podemos e devemos tirar dúvidas entre todos a qualquer momento tanto por discord como por whatsapp e afins.
Assim, tendo em conta a diferença de disponibilidade entre todos, talvez faça sentido combinarmos 3 janelas temporais nas quais, à partida, possamos praticamente todos habitualmente, fora certas exceções, claro, sendo que no início de cada semana, nas aulas, combinamos entre todos uma das três janelas para nos reunirmos nessa semana. 
Porém, como será, naturalmente, difícil reunirmos sempre todos em todas as semanas, é importante garantir que pelo menos 1 ou 2 elementos de cada parte consegue comparecer nessa reunião para dar feedback sobre a respetiva componente, incluindo dúvidas, se surgirem.

Para conseguirmos estabelecer estas 3 janelas, ficará registado abaixo a disponibilidade de cada elemento (preenchi a minha e peço que preencham a vossa).

Prata: Qualquer hora de sábado. Qualquer hora de domingo a partir das 14h30m.
Vasco:
Isadora:
Pedro:
Stefane:
