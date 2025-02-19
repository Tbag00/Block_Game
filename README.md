# Block_Game
Intelligenza artificiale sviluppata con Python che risolve il problema del Blocks World

# Sviluppatori
Trotta Elia, Alessandro Bevilacqua, Bagiana Tommaso

# Intro
Dei blocchi numerati sono impilati in varie colonne su un tavolo,
l' AI deve essere in grado di partire da una configurazione di blocchi e muovere un braccio meccanico per raggiungerne un' altra datagli in input.

# Dominio e vincoli
1. I blocchi possono essere al massimo 6.
2. Non ci sono due blocchi con lo stesso numero
3. Un blocco può essere posizionato in una riga r>0 se e solo se c'e' un altro blocco immediatamente al di sotto.
4. Il braccio meccanico puo' afferrare un blocco alla volta
5. Un blocco non puo' avere piu' di un blocco sopra di lui
6. Il braccio puo' prendere solo il blocco piu' in alto in una colonna e piazzarlo in cima a un' altra colonna

# Azioni (da formalizzare)
Il braccio prende il blocco dalla colonna i e lo piazza in cima alla colonna j, le azioni sono le duple (i, j) con 0 <= i,j <= 10
Non si possono prendere blocchi da una colonna vuota

# Stati
In una matrice 6x6 un valore è 0 se non è occupato da blocchi, i se è occupato dal blocco numero i, non possono esserci due caselle con lo stesso numero

# Euristica
L'euristica utiizzata si basa sul concetto di relaxed problem, conta semplicemente quanti sono i blocchi nella posizione sbagliata.
Il problema di questa euristica è che in alcuni casi impiega discreto tempo essendo molto semplice, quindi abbiamo sviluppato anche una relaxed pesata che conta invece i blocchi sopra i blocchi sbagliati e i blocchi sotto quelli giusti ma non è ammissibile.
Abbiamo fatto anche un euristica basata sul concetto di distanza di manhattan, anch'essa inammissibile ma non per un problema di implementazione ma perchè non rappresentatitva del problema.
Inoltre abbiamo fatto anche un'euristica basata sui subgoal del problema ossia sistemare le colonne, che da quindi la priorita a sistemare la colonna più a sinistra contenente blocchi nel posto sbagliato. Anch'essa non è ammissibile ma mediamente è la più veloce.

# Animazione (da scrivere)

## TODO
-Acquisizione dati in input. Tramite reti neurali e convolutive
-Classificazione dati in input.
-Traduzione dell' input in stati
-IDA*
