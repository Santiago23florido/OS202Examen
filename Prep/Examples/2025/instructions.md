# Examen machine OS 202 du 18 Mars 2025

- Tous les documents sont autorisés.
- **A rendre** : Les trois fichiers pythons parallélisés et un document (pdf, markdown, etc.) pour les réponses aux questions.
- Les fichiers devront être envoyés sous forme de fichiers pythons *non compressés* aux adresses suivantes (selon votre groupe de TP) :
  - xavier.juvigny@onera.fr (Groupe 1)
  - jean-didier.garaud@onera.fr (Groupe 2)
  - apf@crans.org (Groupe 3)


Bon courage !

## A] Traitement de l'image et convolution

Les trois scripts python à paralléliser se reposent sur le même type d'algorithme.

Dans les trois scripts, on cherche à doubler la taille des images (issues de vidéo dans le premier cas, d'une photo dans les deux autres cas) en évitant d'obtenir une version trop pixellisée de l'image finale.

Pour cela, après avoir chargé l'image en RGB (Rouge-Vert-Bleu), on effectue une transformation de l'espace de couleur pour passer dans l'espace de couleur HSV ( Hue (Teinte) - Saturation - Value ) où les composantes H et S permettent de définir une couleur et V l'intensité
lumineuse du pixel (équivalent à un niveau de gris).

L'idée va être de lisser l'image à l'aide d'un flou gaussien sur les trois composantes (H, S et V) de l'image, puis effectuer un filtre de
netteté uniquement sur la composante V. L'idée est que l'œil humain voit avec précision les niveaux de gris mais très grossièrement les couleurs (le cerveau faisant ensuite un traitement pour composer *in fine* une image en couleur précise !)

Pour ces deux filtres, on va utiliser des convolutions discrètes en 2D. Une convolution discrète en 2D est une généralisation de la convolution
en 1D et consiste à avoir pour fonction de convolution une matrice $F$ de dimension $(2m+1)\times (2n+1)$ dont les indices pour les lignes et les colonnes commencent à $-m$ et finissent en $+m$ pour les lignes et $-n$ à $n$ pour les colonnes. Le champ à convoler sera une grille 2D $G$ ayant pour valeurs $g_{ij}$ pour la ième ligne et jème colonne. Les conditions limites seront prises comme une "répétition des valeurs au bord du domaine" (correspond à l'argument `mode='same'` de la fonction `signal.convolve2d`).

La grille $C = F\star G$ convolée sera obtenue pour la valeur $c_{ij}$ se trouvant à la ième ligne et jème colonne par :

$$
C_{i,j} = \sum_{k=-m}^{+m}\sum_{l=-n}^{+n}F_{k,l}.G_{i+k,j+l}
$$

Pour le filtre Gaussien, on choisit pour matrice de convolution la matrice :

$$
F_{G} = \frac{1}{16}\left(\begin{array}{ccc}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{array}\right)
$$

Pour le filtre de netteté, on choisit la matrice de convolution suivante :

$$
F_{S} = \left(\begin{array}{ccc}
 0 & -1 &  0 \\
-1 & 5  & -1 \\
 0 & -1 &  0 \end{array}\right)
$$

Dans le troisième exercice, on appliquera toujours le filtre gaussien pour les composantes H et S de l'image, mais on n'appliquera qu'un seul filtre à la composante V, permettant à la fois de faire un lissage et un filtre de netteté. Ce filtre de dimension $5\times 5$ est le suivant :

$$
F_{D} = \frac{1}{256}\left(\begin{array}{ccccc}
1 &  4 & 6 &  4 &  1 \\
4 & 16 & 24& 16 &  4 \\
6 & 24 & -476 & 24 & 6 \\
4 & 16 & 2    & 16 & 4 \\
1 &  4 & 6 & 4 & 1
\end{array}\right)
$$

## B] À FAIRE

Les codes seront parallélisés en utilisant MPI.

### 1. Environnement de calcul

Donner le nombre de cœurs logiques contenus sur votre machine ainsi que la taille des mémoires caches L1, L2 et L3.

### 2. Parallélisation d'images issues d'une vidéo

Le but de ce programme est de doubler la taille des images d'une vidéo sans trop pixelliser les images agrandies.

On va donc appliquer les filtres $F_{G}$ et $F_{S}$ vus plus haut sur un grand nombre d'images (les images issues de la vidéo). 

- Expliquer votre stratégie de parallélisation et pourquoi ce type de parallélisation est bien adapté à ce problème et est optimal.
- Paralléliser le programme `movie_filter.py` correspondant à ce problème.
- Calculer en fonction du nombre de processus utilisé (dans la limite du nombre de coeurs dont vous disposez) la courbe d'accélération de votre programme parallèle.

Le répertoire `sorties/perroquets` contient un fichier `perroquets.md5sum`. Celui-ci peut être utilisé pour vérifier que votre parallélisation n'a pas modifié les images résultats :

	cd sorties/perroquets
	md5sum -c perroquets.md5sum


### 3. Parallélisation d'une photo en haute résolution (1)

Le but de ce programme est de doubler la taille d'une photo tout en évitant d'avoir une photo pixellisée à la fin.

On cherche à ce que chaque processus utilise un minimum de mémoire en évitant de prendre des bouts d'images trop grands par processus (uniquement le strict minimum nécessaire pour la parallélisation).

On applique ici consécutivement les filtres $F_{G}$ sur les trois composantes H, S et V de l'image et $F_{S}$ sur la composante $V$ de l'image.

- Expliquer votre stratégie de parallélisation et pourquoi ce type de parallélisation est adapté à votre problème.
- Est-ce que cette stratégie mise en place serait optimale pour la parallélisation que vous avez effectuée en 2.
- Paralléliser le programme `double_size.py` correspondant à ce problème.
- Calculer en fonction du nombre de processus utilisé (dans la limite du nombre de coeurs dont vous disposez) la courbe d'accélération de votre programme parallèle.

### 4. Parallélisation d'une photo en haute résolution (2)

Le but de ce programme est de doubler la taille d'une photo tout en évitant d'avoir une photo pixellisée à la fin.

On cherche à ce que chaque processus utilise un minimum de mémoire en évitant de prendre des bouts d'images trop grand par processus (uniquement le strict minimum nécessaire pour la parallélisation).

On applique le filtre $F_{G}$ sur les deux composantes $H$ et $S$ de l'image et le filtre $F_{D}$ sur la composante $V$ de l'image.

- Expliquer votre stratégie de parallélisation et pourquoi ce type de parallélisation est adapté à votre problème.
- Quelles sont les différences, désavantages et avantages de votre stratégie de parallélisation par rapport à la question précédente ?
- Paralléliser le programme `double_size2.py` correspondant à ce problème.
- Calculer en fonction du nombre de processus utilisé (dans la limite du nombre de coeurs dont vous disposez) la courbe d'accélération de votre programme parallèle.