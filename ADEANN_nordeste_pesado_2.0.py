import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from keras.layers import LSTM, Dense, LeakyReLU
from keras.models import Sequential
from keras.activations import relu
from sklearn.model_selection import train_test_split
import numpy as np
import os
import math

import math as mt

# ALGORITMO GENÉTICO
GERACAO = 5
INDIVIDUOS = 10
GENE = 516
NINTER = 10000

# REDE NEURAL
TIPO = 4
N = 4
NINT = 4
NINT1 = 7
NINT2 = 7
NENT = 5
NSAI = 1
NAPD = 900
TARP3 = 0.01
TARP4 = 0.01
MAXITER = 10  # Numero de Epocas
ACEITAVEL = 0.001
CONTID = 0
CONTREGRASVAL = 0
INDIC = 0
SUP = 0
NEURIT = 0

# TAXAS
TX_TRANS = 0.001
ERR = 0.0
EMQ = 0.0
FITNESS = 0.0
AUX = 0
SOMA_NINT_TOTAL = 0
LIMIAR = 0.5

# CONSTANTES
FIT = 0
HIST_FIT = []
for i in range(0, GERACAO):
    HIST_FIT.append([])
# HIST_FIT.shape = (GERACAO, 1)

tab_converte = ['f', 'F', 'n', '.', 'n', '.', 'f', 'F', 'F', 'f', 'B', 'f', '[', 'n', '[', '.',
                'f', ']', 'n', '*', '.', 'F', 'f', 'F', ']', '.', '[', 'f', 'f', '*', 'B', ']',
                '.', ']', 'n', 'F', 'f', 'B', 'f', 'B', 'F', '[', 'B', 'n', '*', 'f', '.', ']',
                ']', '[', 'n', 'F', 'n', 'B', '[', '.', 'f', ']', 'B', 'F', 'B', 'f', '*', '[']

pFile = open("relat_nordeste_pesado.txt", "w")




data = pd.read_csv('dados1.csv', encoding='utf-8', delimiter=';')
data_y =  data['Pesado'] #+ data['Leve'])/3
data_x = data.drop(["Pesado", "Medio", "Leve"], axis=1)
data_x = data_x.drop(["DiaInicio", "DiaFinal", "MesInicio", "AnoInicio", "MesFinal", "AnoFinal"], axis=1)

steps = 36 ########################################################
_type = 'heavy' # 'heavy' #'light'
_location = 'northeast'
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=steps/len(data_x),  random_state=42)



def zerar_fitness(fit):
    fit = np.zeros(INDIVIDUOS * GENE)
    fit.shape = (INDIVIDUOS, GENE)
    return fit


def imprime_hist_fitness(hist_fit, file, contador):
    # imprimir no txt
    print("\n", contador)
    print("\t\t", hist_fit[contador - 1][0])
    print("\t\t", hist_fit[contador - 1][1])
    pFile.write("\n"+str(contador))
    pFile.write("\t\t"+ str(hist_fit[contador - 1][0]))
    pFile.write("\t\t"+ str(hist_fit[contador - 1][1]))


def genotipo_estatico(individuos, gene):
    gen = np.zeros(individuos * gene)
    gen.shape = (individuos, gene)
    for i in range(0, gene):
        for j in range(0, individuos):
            gen[i][j] = np.random.random_integers(0, 1, 1)[0]
    return gen


def genotipo_dinamico(individuos, gene):
    print(gene, " - ", int(gene))
    gene = int(gene)
    gen = np.zeros(individuos * gene)
    gen.shape = (individuos, gene)

    for i in range(0, individuos):
        for j in range(0, gene):
            gen[i][j] = np.random.random_integers(0, 1, 1)[0]

    return gen


def gen_bin_dec_genotipo_dinamico(individuos, gene_dec):
    gen_dec = np.zeros(individuos * gene_dec)
    gen_dec.shape = (individuos, gene_dec)
    return gen_dec


def genotipo_dinamico_string(individuos, gene_dec):
    print(gene_dec)
    gene_dec = int(gene_dec)
    gen_string = np.chararray(individuos * (gene_dec + 1))
    gen_string.shape = (individuos, gene_dec + 1)
    return gen_string


def imprime_genbin(gen, individuos, gene, file):
    print("\n")
    for i in range(0, individuos):
        for j in range(0, gene):
            #pFile.write(str(gen[i][j])) <<-------- zeros e uns
            print(gen[i][j])
    return gen


def imprime_genbindec(gen_bin_dec, individuos, gene_dec):
    print("\n")
    gene_dec = int(gene_dec)
    for i in range(0, individuos):
        print("-Individuo[" + str(i + 1) + "]-\n")

        for j in range(0, gene_dec):
            print("\t", gen_bin_dec[i][j])
    print("\n\n\n")
    return gen_bin_dec


def imprime_genstring(gen_string, individuos, gene_dec, file):
    global CONTID, INDIC, FIT
    for i in range(0, individuos):
        print("\n-Individuo[" + str(i + 1) + "]-\n")
        pFile.write("\n-Individuo["+str(i + 1)+"]-\n")
        for j in range(0, gene_dec):
            print(gen_string[i][j])
            pFile.write(str(gen_string[i][j]))
        CONTID += 1
        if gen_string[i][gene_dec] is b'V':

            print("\t STRING VALIDA")
            pFile.write("\t STRING VALIDA")
            mapeamento_genotipo_fenotipo(NENT, NSAI, 0, TIPO, file)  # 0 -> aleatorio
        else:
            FIT[INDIC][0] = INDIC + 1
            FIT[INDIC][1] = 0.0
            FIT[INDIC][2] = FIT[INDIC][0]
            FIT[INDIC][3] = INDIC + 1
            if INDIC >= 1:
                FIT[INDIC][3] = INDIC + 1 + FIT[INDIC - 1][3]
            else:
                FIT[INDIC][3] = INDIC + 1
            FIT[INDIC][4] = 0.0
            FIT[INDIC][5] = 0.0
            FIT[INDIC][6] = 0.0
            INDIC += 1
            print("\t STRING INVALIDA")
            pFile.write("\t STRING INVALIDA")
    return FIT


def legenes_genbin(gen, gen_bin_dec, individuos, gene):
    # gene1, gene2, gene3, gene4, gene5, gene6, i;
    start = 0
    j = 0
    compactador = 0
    stop = int(gene / 6)

    while j < individuos:
        for i in range(0, stop):
            gene1 = gen[j][start]
            gene2 = gen[j][start + 1]
            gene3 = gen[j][start + 2]
            gene4 = gen[j][start + 3]
            gene5 = gen[j][start + 4]
            gene6 = gen[j][start + 5]
            decimal = converte_genbindec(gene1, gene2, gene3, gene4, gene5, gene6, j, compactador)
            gen_bin_dec[j][compactador] = decimal
            compactador += 1
            start = start + 6
        j += 1
        start = 0
        compactador = 0


def converte_genbindec(gene1, gene2, gene3, gene4, gene5, gene6, j, compactador):
    decimal = gene6 * 32 + gene5 * 16 + gene4 * 8 + gene3 * 4 + gene2 * 2 + gene1
    return decimal


def legenes_genbindec_string(gen_bin_dec, gen_string, individuos, gene_dec):
    for j in range(0, individuos):
        for i in range(0, gene_dec):
            gen_string[j][i] = tab_converte[int(gen_bin_dec[j][i])]

    return gen_string


def avalia_regras_gen_string(gen_string, individuos, gene_dec):
    string_val = [b'.', b'f', b'[', b'F', b'f', b'n', b'B', b']']
    for j in range(0, individuos - 1):
        for i in range(0, gene_dec - 7):
            # Le do inicio para o final com um passo de um bit.
            if (gen_string[j][i] is b'.') and (gen_string[j][i + 1] is b'f') and (gen_string[j][i + 2] is b'[') and (
                    gen_string[j][i + 3] is b'F') and (gen_string[j][i + 4] is b'f') and (
                    gen_string[j][i + 5] is b'n') and (gen_string[j][i + 6] is b'B') and (gen_string[j][i + 7] is b']'):
                gen_string[j][gene_dec] = b'V'

    for j in range(0, individuos - 1):
        # Le do final para o inicio com um passo de um bit.
        for i in range(gene_dec - 1, 6, -1):
            if (gen_string[j][i] is b'.') and (gen_string[j][i + 1] is b'f') and (gen_string[j][i + 2] is b'[') and (
                    gen_string[j][i + 3] is b'F') and (gen_string[j][i + 4] is b'f') and (
                    gen_string[j][i + 5] is b'n') and (gen_string[j][i + 6] is b'B') and (gen_string[j][i + 7] is b']'):
                gen_string[j][gene_dec] = b'V'

    # mudei aqui baixo todo la?o for

    for j in range(0, individuos - 1):
        # Le de forma sequencial bit a bit do inicio para o fim.
        cont_elem = 0
        pos_string = 0

        for i in range(0, gene_dec - 1):
            carac = gen_string[j][i]

            if not (cont_elem >= 8):
                if string_val[pos_string] is carac:
                    cont_elem += 1
                    pos_string += 1

            if cont_elem is 8:
                gen_string[j][gene_dec] = b'V'
                # //printf("j=%d",j);
                # //printf("cont_elem=%d",cont_elem);
                # //getch();
    # print("\n2222222222222\n")
    # print(gen_string)
    return gen_string


def mapeamento_genotipo_fenotipo(NENT, NSAI, aleatorio, TIPO, file):
    global NINT, NINT1, NINT2
    # if NR is NR1:
    #     NR1 = np.random.random_integers(2, 20)
    N_TIPO = np.random.random_integers(10, 50, 1)[0]

    NINT = N_TIPO
    NINT1 = N_TIPO
    NINT2 = np.random.random_integers(10, 50, 1)[0]

    NINT_N3 = np.zeros(1)
    NINT_N3[0] = NINT1
    SIZE_N3 = len(NINT_N3)

    NINT_N4 = np.zeros(2)
    NINT_N4[0] = NINT1
    NINT_N4[1] = NINT2
    SIZE_N4 = len(NINT_N4)
    ############################################################################################
    # n == 3 ? Mapeamento(NENT, NSAI, NINT_N3, SIZE_N3, "1.4", pFile) : Mapeamento(NENT, NSAI, NINT_N4, SIZE_N4, "1.4", pFile); // (ENTRADA, SAIDA, NR, TIPO)

    try:
        cmd = "c_exec\\executavel.exe "+str(NENT)+" "+str(NSAI)+" "+str(N)+" "+str(NINT1)+" "+str(NINT2)+" "
        cmd_out = os.popen(cmd).read()
        pFile.write(str(cmd_out))
    except TypeError:
        print("\n\n ERROR_CMD_TYPE \n\n")
        exit(0)

    if N is 3:
        treina_rede(CONTID, file, NINT)
    if N is 4:
        treina_rede_(CONTID, file, NINT1, NINT2)


def NR_RAND(int):
    pass


def treina_rede(individuos, file, NINT):
    # arr = np.zeros(9)
    # arr.shape = (3,3)
    # print(arr)
    # from keras.models import Sequential
    # from keras.layers import Dense

    # model = Sequential()
    # model.add(Dense(units=64,activation='relu', input_dim=2))
    # model.add(Dense(units=10, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])

    # # model.fit(x_train, y_train, epochs=5, batch_size=32)
    print("Treina rede")


def treina_rede_(contind, file, NINT1, NINT2):
    global EMQ, FITNESS, INDIC, LIMIAR, NENT, NINT, NSAI, N, MAXITER, err, total, acertos
    print("\nNENT=(" + str(NENT - 1) + " Entradas + 1 Bias)")
    print("\nNINT1=(" + str(NINT1 - 1) + " Int + 1 Bias)")
    print("\nNINT2=(" + str(NINT2 - 1) + " Int + 1 Bias)")
    print("\nNSAI=", NSAI)
    print("\n\nTreinamento do Individuo=", contind)

    pFile.write("\nNENT=(" + str(NENT - 1) + " Entradas + 1 Bias)")
    pFile.write("\nNINT1=(" + str(NINT1 - 1) + " Int + 1 Bias)")
    pFile.write("\nNINT2=(" + str(NINT2 - 1) + " Int + 1 Bias)")
    pFile.write("\nNSAI="+str(NSAI))
    pFile.write("\n\nTreinamento do Individuo="+str(contind))

    limiar = 0.1
    lrelu = lambda x: relu(x, alpha=0.3)
    model = Sequential()
    model.add(Dense(units=NINT1, activation='relu', input_dim=5))
    model.add(Dense(units=NINT2, activation=lrelu))
    model.add(Dense(units=1, activation=lrelu))

    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=['mean_absolute_error'])

    model.fit(x_train, y_train, epochs=10000, batch_size=20, verbose=0)

    predictions = model.predict(x_test)

    total = len(x_test)
    acertos = 0
    rounded = [x for x in predictions]  # [np.round(x) for x in predictions]

    EMQ = 0.0
    MAE = 0.0
    MAPE = 0.0

    for i, predic in enumerate(rounded):

        err = (0.5 * math.fabs((y_test.iloc[i] - predic) ** 2)) ** (1 / 2)
        mae = (0.5 * math.fabs(y_test.iloc[i] - predic))
        mape = 0.5 * (math.fabs(y_test.iloc[i] - predic) / np.mean(y_test))

        print("\nPadrao>>", i)
        print("\ncalculado>>" + str(predic) + "   \tdesejado>>" + str(y_test.iloc[i]) + "  \tEmq>>", err, "\tMae>>", mae,
              "\tMape>>", mape)
        pFile.write("\nPadrao>>" + str(i))
        pFile.write("\ncalculado>>" + str(predic) + "  \tdesejado>>" + str(y_test.iloc[i]) + "  \tEmq>>"+str( err)+ "\tMae>>"+str(
                    mae)+
                    "\tMape>>" + str(mape))

        EMQ += err
        MAE += mae
        MAPE += mape

        if predic <= (y_test.iloc[i] + y_test.iloc[i] * limiar) and predic >= (y_test.iloc[i] - y_test.iloc[i] * limiar):
            acertos += 1

    EMQ = EMQ / len(rounded)
    MAE = MAE / len(rounded)
    MAPE = MAPE / len(rounded)

    y_wrapper = y_test.reset_index(drop=True)

    FITNESS = (acertos / total)

    print("\nemq>> ", EMQ)
    print("\nmae>> ", MAE)
    print("\nmape>> ", MAPE)
    print("\nfitness>> ", FITNESS)
    print("\n\n<<Pesos Camada Entrada Oculta>>\n")
    pFile.write("\nemq>> " + str(EMQ))
    pFile.write("\nmae>> " + str(MAE))
    pFile.write("\nmape>> "+ str(MAPE))
    pFile.write("\nfitness>> " + str(FITNESS))
    pFile.write("\n\n<<Pesos Camada Entrada Oculta>>\n")




    predictions_full = model.predict(data_x)
    plt.figure(figsize=(16, 7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(data_y, color="#2ca02c", linewidth=4)
    plt.plot(predictions_full, color="#1f77b4", linestyle='--', linewidth=4)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,
               handles=[mpatches.Patch(label="Real", color="green"), mpatches.Patch(label="Predict", color="blue")])
    plt.xlabel('Steps')
    plt.title('Total prediction of the subject '+str(contind)+' - '+str(_location)+' '+str(_type))
    plt.tight_layout()
    plt.savefig(_location+'_'+_type+'_total_'+str(contind)+'.png', dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    # plt.show()
    plt.close()

    # TODO salvar em pasta ajustar linhas e limpar cache
    # plt.plot(predictions, color="r", linewidth=1)
    # plt.plot(y_wrapper, color="g", linewidth=1)
    # plt.legend(handles=[mpatches.Patch(label='The red data'), mpatches.Patch(label='The red data2')])
    # plt.savefig('pesado/i_'+str(contind)+'.png', dpi=None, facecolor='w', edgecolor='w',
    # orientation='portrait', papertype=None, format=None,
    # transparent=False, bbox_inches=None, pad_inches=0.1,
    # frameon=None, metadata=None)
    # plt.close()
    plt.figure(figsize=(16, 7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(y_wrapper, color="#2ca02c",  marker="o", linewidth=4, markersize=10 )
    plt.plot(predictions, color="#1f77b4", linestyle='--',  marker="8", linewidth=4,markersize=10)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,
               handles=[mpatches.Patch(label="Real", color="green"), mpatches.Patch(label="Predict", color="blue"),
                        mpatches.Patch(label="RMSE >> " + str(round(EMQ, 5)), color="white"),
                        mpatches.Patch(label="MAE  >> " + str(round(MAE, 5)), color="white"),
                        mpatches.Patch(label="MAPE >> " + str(round(MAPE, 5)), color="white")])
    plt.xlabel('Steps')
    plt.title('Prediction of the subject '+str(contind)+' - '+str(_location)+' '+str(_type))
    plt.tight_layout()
    plt.savefig(_location+'_'+_type+'_prediction_'+str(contind)+'.png', dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()

    index = np.arange(1)

    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, EMQ, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='RMSE >> ' + str(round(EMQ, 5)))

    rects2 = ax.bar(index + bar_width, MAE, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label='MAE   >> ' + str(round(MAE, 5)))

    rects3 = ax.bar(index + 2 * bar_width, MAPE, bar_width,
                    alpha=opacity, color='g',
                    error_kw=error_config,
                    label='MAPE >> ' + str(round(MAPE, 5)))

    plt.rcParams.update({'font.size': 16})
    ax.set_title('Scores of the Subject '+str(contind)+' - '+str(_location)+' '+str(_type))
    plt.xticks([index, index + bar_width, index + 2 * bar_width],
               ["RMSE", "MAE", "MAPE"])
    ax.legend()

    fig.tight_layout()
    # plt.show()
    plt.tight_layout()
    plt.savefig(_location+'_'+_type+'_score_'+str(contind)+'.png', dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.close()

    EMQ = err

    # antiga finess
    # FITNESS = 1000 * (exp(-emq) * exp(-NINT)) + (1 / (emq * NINT));

    # nova fitness
    FITNESS = (acertos / total)
    FIT[INDIC][0] = INDIC + 1
    FIT[INDIC][1] = FITNESS
    FIT[INDIC][2] = FIT[INDIC][0]
    FIT[INDIC][3] = INDIC + 1
    if INDIC + 1 > 1:
        FIT[INDIC][3] = INDIC + 1 + FIT[INDIC - 1][3]
    else:
        FIT[INDIC][3] = INDIC + 1
    FIT[INDIC][4] = EMQ
    FIT[INDIC][5] = NINT
    FIT[INDIC][6] = NINT1 + NINT2 + NENT + NSAI
    if N is 4:
        FIT[INDIC][7] = NINT2

    INDIC += 1
    print("\nemq>> ", EMQ)
    print("\nfitness>> ", FITNESS)
    print("\n\n<<Pesos Camada Entrada Oculta>>\n")
    pFile.write("\nemq>> " + str(EMQ))
    pFile.write("\nfitness>> " + str(FITNESS))
    pFile.write("\n\n<<Pesos Camada Entrada Oculta>>\n")

    # melhorar o resultado abaixo -------------------------------------------------------------------
    print(str(model.get_weights()))
    pFile.write(str(model.get_weights()))


def zera_fitness(fit):
    fit = np.zeros(INDIVIDUOS * GENE)
    fit.shape = (INDIVIDUOS, GENE)
    return fit


def ordena(FIT, file):
    global m
    m = 0
    for i in range(0, INDIVIDUOS - 1):
        m = i
        for j in range(i + 1, INDIVIDUOS):
            aux1 = FIT[j][1]
            aux2 = FIT[m][1]
            if aux1 > aux2:
                m = j

        ch = FIT[i][0]
        ch1 = FIT[i][1]
        ch2 = FIT[i][4]
        ch3 = FIT[i][5]
        ch4 = FIT[i][6]
        ch5 = FIT[m][0]
        FIT[i][0] = ch5
        FIT[i][1] = FIT[m][1]
        FIT[i][4] = FIT[m][4]
        FIT[i][5] = FIT[m][5]
        FIT[i][6] = FIT[m][6]
        FIT[m][0] = ch
        FIT[m][1] = ch1
        FIT[m][4] = ch2
        FIT[m][5] = ch3
        FIT[m][6] = ch4


def main():
    file = ''
    global FIT, HIST_FIT, MAXITER, INDIC
    # pFile = fopen("relat_autoMaasd2asdasd.txt", "w");
    for n in range(4, 4):

        if (n is 3):
            pass
            # pFile = fopen("relat_N3asdas.txt", "w");
        else:
            pass
            # fprintf(pFile, "\nTARP1 %.2f\nTARP2 %.2f\n\n\n");

    # gene : represneta o numero de genes por inidividuo, nesse caso 516;
    gene_dec = int(GENE / 6)  # //43+1=44 ultimo elemento armazena status da string valida % = valida;

    # //////    printf("Gerando Genotipo Aleatoriamente!\n");
    gen_bin = genotipo_dinamico(INDIVIDUOS, GENE)
    gen_bin_dec = genotipo_dinamico(INDIVIDUOS, gene_dec)
    gen_string = genotipo_dinamico_string(INDIVIDUOS, gene_dec)

    FIT = zera_fitness(FIT)
    contador = 1  # //contador do numero de geracoes
    global contador1
    contador1 = 0
    # //int pontos_corte=(gene_dec*0.9);

    while contador <= GERACAO:

        imprime_genbin(gen_bin, INDIVIDUOS, GENE, file)
        legenes_genbin(gen_bin, gen_bin_dec, INDIVIDUOS, GENE)
        imprime_genbindec(gen_bin_dec, INDIVIDUOS, gene_dec)
        legenes_genbindec_string(gen_bin_dec, gen_string, INDIVIDUOS, gene_dec)
        contador1 = 1  # //contador do sumero de cruzamentos por gera??o
        FIT = zera_fitness(FIT)
        gen_string = avalia_regras_gen_string(gen_string, INDIVIDUOS, gene_dec)  # //Avalia Regras Validas
        imprime_genstring(gen_string, INDIVIDUOS, gene_dec, file)  # //Imprime Individuo[i] e DNA
        # //Ordenar Individuos
        ordena(FIT, file)
        # //Imprimir Individuos

        imprime_fitness(FIT, file, contador)
        contador += 1
        INDIC = 0
        while contador1 <= (gene_dec * 0.8):
            # realiza (individuos/2) cruzamentos
            gen_bin = selecao(gen_bin, gen_string, gene_dec)
            contador1 += 1

        contador1 = 1
        while contador1 <= (gene_dec * 0.8):
            # //516-86-14
            gen_bin = mutacao(gen_bin)
            contador1 += 1
        # // transloca(gen_string,gene_dec);

        if GERACAO < 20:
            MAXITER = MAXITER + 5
        elif (GERACAO >= 20) and (GERACAO < 40):
            MAXITER = MAXITER + 10
        else:
            MAXITER = MAXITER + 2

    contador = 1

    imprime_cabec(file)
    while GERACAO >= contador:
        imprime_hist_fitness(HIST_FIT, file, contador)
        contador += 1

    # //End Do

    print("\n\n<<Simulacao Concluida - Relatorio Gerado!!>>")
    pFile.write("\nsimulacao concluida")


def selecao(gen, gen_string, gene_dec):
    global FIT, e5

    e1 = 0
    e = 0
    j = INDIVIDUOS - 1

    while e == e1:
        e = np.random.random_integers(0, 1, 1)[0]
        e2 = int(FIT[e][1])

        e2 -= 1
        if e2 >= (INDIVIDUOS / 2):
            e2 = 1
        e1 = np.random.random_integers(0, 1, 1)[0]
        e3 = int(FIT[e1][1])
        e3 -= 1
        if e3 >= (INDIVIDUOS / 2):
            e3 = 0

    e4 = np.random.random_integers(0, 1, 1)[0]
    e5 = int(FIT[e4][1])
    e5 -= 1
    if e5 <= (INDIVIDUOS / 2) - 1:
        e5 = j - e5

    return cruzamento(e2, e3, e3, gen, gen_string, gene_dec)


def cruzamento(i, j, pos, gen, gen_string, gene_dec):
    pc = np.random.random_integers(0, INDIVIDUOS, 1)[0]
    pc = (pc / 100)
    ale = (GENE - 12)
    gen_ale = np.random.random_integers(0, ale, 1)[0]
    gen_ale1 = np.random.random_integers(0, ale, 1)[0]
    if pc <= 0.6:
        gen[pos][gen_ale] = gen[i][gen_ale]
        gen[pos][gen_ale1] = gen[j][gen_ale1]
        gen[pos][gen_ale + 1] = gen[i][gen_ale + 1]
        gen[pos][gen_ale1 + 1] = gen[j][gen_ale1 + 1]
        # //
        gen[pos][gen_ale + 2] = gen[i][gen_ale + 2]
        gen[pos][gen_ale1 + 2] = gen[j][gen_ale1 + 2]
        # //
        gen[pos][gen_ale + 3] = gen[i][gen_ale + 3]
        gen[pos][gen_ale1 + 3] = gen[j][gen_ale1 + 3]
        # //
        gen[pos][gen_ale + 4] = gen[i][gen_ale + 4]
        gen[pos][gen_ale1 + 4] = gen[j][gen_ale1 + 4]
        # //
        gen[pos][gen_ale + 5] = gen[i][gen_ale + 5]
        gen[pos][gen_ale1 + 5] = gen[j][gen_ale1 + 5]
    return gen


def mutacao(gen):
    pm = np.random.random_integers(0, 10, 1)[0]
    pm = (pm / 100)

    e6 = np.random.random_integers(0, 1, 1)[0]
    e7 = int(FIT[e6][0])
    e7 -= 1
    if e7 <= (INDIVIDUOS / 2) - 1:
        e7 = (INDIVIDUOS - 1) - e6

    ale = (GENE - 1)
    gen_ale = np.random.random_integers(0, ale, 1)[0]

    if (gen[e7][gen_ale] is 0) and (pm <= 0.1):
        gen[e7][gen_ale] = 1
    else:
        gen[e7][gen_ale] = 0
    return gen


def imprime_fitness(fit, file, contador):
    global HIST_FIT, SOMA_NINT_TOTAL, SUP, N, soma_fit, soma_nint, contador_val, var_nint
    soma_fit = 0.0
    soma_nint = 0.0
    contador_val = 0
    var_nint = 0.0

    pFile.write("\n\nResumo da Geracao:"+str(contador))
    pFile.write("\n_______________________________________________________________")
    pFile.write("\nIND  |Fitness      |Posto  |Acum    |EMQ      |NINT  |NT") if N == 3 else pFile.write("\nIND  |Fitness      |Posto  |Acum    |EMQ      |NINT1 |NINT2  |NT")
    pFile.write("\n_______________________________________________________________")
    print("\n\nResumo da Geracao: ", contador)
    print("\n_______________________________________________________________")
    print(
        "\nIND  |Fitness      |Posto  |Acum    |EMQ      |NINT  |NT" if N == 3 else "\nIND  |Fitness      |Posto  |Acum    |EMQ      |NINT1 |NINT2  |NT")
    print("\n_______________________________________________________________")

    for i in range(0, INDIVIDUOS):

        print("\n", fit[i][0])
        print("    ", fit[i][1])
        print("     ", fit[i][2])
        print("     ", fit[i][3])
        print("      ", fit[i][4])
        print("   ", fit[i][5])
        pFile.write("\n"+str(fit[i][0]))
        pFile.write("    "+str(fit[i][1]))
        pFile.write("     "+str(fit[i][2]))
        pFile.write("     "+str(fit[i][3]))
        pFile.write("      "+str(fit[i][4]))
        pFile.write("   "+str(fit[i][5]))
        if N == 4:
            print("      ", fit[i][7])
            pFile.write("      "+str(fit[i][7]))
        print("   ", fit[i][6])
        pFile.write("   "+str(fit[i][6]))
        soma_fit = soma_fit + fit[i][1]
        if 0 < fit[i][5]:
            soma_nint = soma_nint + fit[i][5]
            contador_val += 1
        SUP = fit[i][3]

    print("\n===============================================================\n")
    pFile.write("\n===============================================================\n")
    if 0 < contador_val:
        SOMA_NINT_TOTAL = SOMA_NINT_TOTAL + (soma_nint / contador_val)
        print("\nMedia do Fitness=\n", soma_fit / contador_val)
        print("Media de Neuronios na Camada Intermediaria na Geracao=\n", soma_nint / contador_val)
        pFile.write("\nMedia do Fitness="+str(soma_fit / contador_val)+"\n")
        pFile.write("Media de Neuronios na Camada Intermediaria na Geracao="+ str(soma_nint / contador_val)+"\n")
    # [(None, []), (None, []), (None, []), (None, []), (None, []), (None, []), (None, []), (None, []), (None, []), (None, [])]

    HIST_FIT[contador - 1].append(("fitness", fit[0][1]))

    # HIST_FIT[contador - 1][0] = fit[0][1];

    if 0 < contador_val:
        # HIST_FIT[contador - 1][1] = soma_fit / contador_val;
        HIST_FIT[contador - 1].append(("fitness_medio", soma_fit / contador_val))
    else:
        # HIST_FIT[contador - 1][1] = 0.0;
        HIST_FIT[contador - 1].append(("fitness_medio", 0.0))

    for i in range(0, INDIVIDUOS):

        if 0 < fit[i][5]:
            var_nint = var_nint + pow((fit[i][5] - (soma_nint / contador_val)), 2) / contador_val

    print("Variancia de Neuronios na Camada Intermediaria na Geracao=\n", var_nint)
    print("Desvio Padrao de Neuronios na Camada Intermediaria na Geracao=\n", (var_nint) ** (1 / 2))
    pFile.write("Variancia de Neuronios na Camada Intermediaria na Geracao="+str(var_nint)+"\n")
    pFile.write("Desvio Padrao de Neuronios na Camada Intermediaria na Geracao="+ str((var_nint) ** (1 / 2))+"\n")
    if 0 < contador_val:
        print("Erro Padrao de Neuronios na Camada Intermediaria na Geracao=\n",
              var_nint ** (-1 / 2) / contador_val ** (1 / 2))
        print("Coeficiente de Variacao de Neuronios na Camada Intermediaria na Geracao=",
              var_nint ** (1 / 2) / (soma_nint / contador_val), "\n")

        pFile.write("Erro Padrao de Neuronios na Camada Intermediaria na Geracao="+str(((var_nint) ** (1 / 2)) / ((contador_val)**(1/2)))+"\n")
        pFile.write("Coeficiente de Variacao de Neuronios na Camada Intermediaria na Geracao="+str(((var_nint)**(1/2)) / (soma_nint / contador_val))+"\n")

    if contador == GERACAO:
        pFile.write("\n\nMedia de Neuronios na Camada Intermediaria na Simulacao="+str((SOMA_NINT_TOTAL / contador))+"\n")
        print("\n\nMedia de Neuronios na Camada Intermediaria na Simulacao=", (SOMA_NINT_TOTAL / contador), "\n")


def imprime_cabec(file):
    print("\nPercentagem de Regras Validas=", (CONTREGRASVAL / (INDIVIDUOS * GERACAO)) * 100.0, "\n")
    print("\n\nHistorico do Fitness na Simulacao")
    print("\n________________________________________________________________")
    print("\n Geracao  |Melhor Fitness         |Fitness Medio")
    print("\n________________________________________________________________")
    pFile.write("\nPercentagem de Regras Validas="+str((CONTREGRASVAL / (INDIVIDUOS * GERACAO)) * 100.0)+"\n")
    pFile.write("\n\nHistorico do Fitness na Simulacao")
    pFile.write("\n________________________________________________________________")
    pFile.write("\n Geracao  |Melhor Fitness         |Fitness Medio")
    pFile.write("\n________________________________________________________________")


main()
