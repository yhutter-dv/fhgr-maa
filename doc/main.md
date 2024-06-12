---
titlepage: true
titlepage-color: "ffffff"
titlepage-rule-color: "ffffff"
titlepage-text-color: "000000"
toc-own-page: true
colorlinks: false
title: Zusammenfassung Methods and Algorithms
author:
  - Yannick Hutter
lang: de
date: "17.02.2024"
lof: true
mainfont: Liberation Sans
sansfont: Liberation Sans
monofont: JetBrains Mono
header-left: "\\small \\thetitle"
header-center: "\\small \\leftmark"
header-right: "\\small \\theauthor"
footer-left: "\\leftmark"
footer-center: ""
footer-right: "\\small Seite \\thepage"
...

\newpage

# Unterrichtsnotizen

## 17.02.2024

### Unterschied zwischen Baum und Graf

Ein Baum hat keinen Zyklus, d.h. man kann nur von oben nach unten wandern aber nicht mehr zurück.

### Unterschied zwischen Simulation und Data Science

Die Datengrundlage für Simulationen wird aufgrund von physikalischen Modellen, Formeln automatisch erzeugt, wohingegen diese für Data Science bereits vorhanden ist.

### Unterschied zwischen K-Nearest Neighbour und K-Means

Im K-Nearest Neighbour wird darauf geachtet, was die "Nachbarn" machen. Auf Grundlage dieser Beobachtung wird dann die eigene Entscheidung getroffen.

### Was ist der Unterschied zwischen einer linearen und einer nicht-linearen Funktion

Eine lineare Funktion entspricht einer Geraden, wohingegen eine nicht-lineare Funktion einer Kurve entspricht. Bei der linearen Funktion kann aufgrund des X-Werte auf den Y-Wert geschlossen werden **und umgekehrt**. Dies ist bei der nicht linearen Funktion nicht der Fall.

### Was bedeutet eine Steigung von 1?

Eine Steigung von 1 entspricht 45%, d.h Delta-X und Delta-Y sind gleich gross

## 24.02.2024

### Was ist der MSE Wert?

Der MSE, d.h. Mean Squared Error beschreibt die Güte der Regressionsgerade. Ist der Wert 0 so liegen die Punkte perfekt auf der Gerade. Ein negativer Wert ist nicht möglich.

> Achtung: Der Wert ist nicht normiert.

### Was ist der R2 Wert?

Der R2 Wert ist ein normierter Wert. Der Wert sagt aus, wie das Verhältnis zischen der erklärten Varianz zur nicht erklärten Varianz ist. 1 bedeutet die Gerade passt perfekt auf die Punkte und 0 bedeutet die Gerade passt überhaupt nicht zu den Punkten.

### Was ist Overfitting?

Das Modell kann genau nur die gelernten Daten vorhersagen aber keine neuen.

### Was ist Underfitting?

Das Modell kann die Daten gar nicht vorhersagen, es ist unbrauchbar evtl. aufgrund einer zu kleinen Datengrundlage.

### Was ist K-Nearest-Neighbour?

Ist ein Standardalgorithmus zur Klassifikation. Der K-Nearest Neighbour ist ein Algorithmus des `supervised Learnings` (d.h. wir kennen die Antwort). Hierzu werden die `ähnlichsten Nachbarn` miteinander verglichen, d.h. es muss ein Mass für "Ähnlichkeit" definiert werden. Folgende Bausteine sind für den Algorithmus notwendig:

- Ähnlichkeits- bzw. Distanzmetrik (definiert was "ähnlich" bedeutet). Hierzu wird oftmals der **Euklidische Abstand** (Luftlinie zwischen zwei Punkten) verwendet.
- Klassifikationsalgorithmus
- Festlegung eines Bewertungsmass (wie gut oder schlecht war die Klassifikation)
- Aufteilung der Daten in Trainings- und Testdaten
- Mehrere Durchläufe für unterschiedliche k-Werte da wir nicht wissen welcher k-Wert optimal ist
- Wählen den k-Wert welcher am besten abgeschnitten hat
- Klassifiziere die Daten anhand des gewählten k-Wertes

Beim K-Nearest-Neighbour können `Skalierungseffekte` auftreten, wenn bspw. die eine Grösse markant grösser ist als die andere, d.h. der eine Wert vom anderen dominiert wird. Um diesem Effekt entgegenzuwirken müssen die Werte auf eine Art und Weise einander angeglichen werden.

Der Algorithmus läuft folgendermassen ab:

- Wenn ein neuer Punkt hinzugefügt wird muss der Abstand zu allen anderen vorhandenen Punkten berechnet werden
- Anschliessend schaut man sich die Abstände in einem bestimmten Radius an und es gilt der Mehrheitsentscheid

> Achtung: Je höher dimensioniert der Raum ist, desto schlechter funktioniert der k-NN Algorithmus, da die Punkte je höher die Dimension wird, weiter auseinanderliegen.

### Was ist der K-Means Algorithmus?

Ist ebenfalls ein Standardalgorithmus zur Klassifikation. Der K-Means Algorithmus ist ein Algorithmus des `unsupervised Learnings` (d.h. wir kennen die Antwort nicht).

### Was besagt die Kardinalität einer Menge?

Die Kardinalität besagt, wie viele Elemente eine Menge beinhaltet.

## 02.03.2024

### Wann kann der K-Means Algorithmus angewendet werden?

Falls wir `unklassifizierte Daten` besitzen, von denen wir die Labels nicht wissen. Man möchte aber trotzdem etwas aus den Daten lernen, d.h. man möchte die Daten trotzdem in bestimmte Gruppen einteilen. Die Komplexität des Algorithmus ist `O(N)`.

### Was ist die Idee hinter dem K-Means Algorithmus?

- Klassifikation der Daten durch Clusterbildung, d.h. unter der Annahme: "Wenn die Daten nahe beieinander liegen, müssen sie eine gewisse Ähnlichkeit besitzen"
- Cluster sollen Elemente `mit ähnlichen Attributen` enthalten
- Der Anwender des Algorithmus muss angeben `wieviele Cluster` man will, d.h. `k wird vorgegeben`
- Der Algorithmus denkt sich `k beliebige Punkte im Raum aus`
- Anschliessend wird der euklidische Abstand (Pythagoras) eines jeden Punktes zu den definierten k's ermittelt
- Die Punkte werden anhand des euklidischen Abstandes den jeweiligen k's zugeordnet
- Anschliessend wird der Schwerpunkt der Cluster berechnet (zur Einfachheit wird der Mittelpunkt der Cluster berechnet)
- Anschliessend werden wieder die Abstände berechnet, bis sich der Mittelpunkt nicht mehr gross ändert (mit Änderung ist die Änderung der Klasse gemeint, nicht die Position)

### Welche Probleme können beim K-Means Algorithmus auftreten?

- Richtige Wahl des k's
- Konvergenz (Erreichung eines finalen Ergebnis) ist nicht garantiert, d.h. Algorithmus muss irgendwann abgebrochen werden (Definition maximale Anzahl von Durchläufen)
- Ob die Klassifikation der Cluster sinnvoll ist muss der Anwender entscheiden
- Der K-Means Algorithmus ist im Vergleich zu anderen Algorithmus zwar sehr einfach aber auch sehr schnell

### Was ist die Trägheit (Inertia)?

Ist die Summe aller Abstände von den einzelnen Punkten zum Cluster Mittelpunkt, d.h. ist die Trägheit sehr gross, sind die Punkte sehr verstreut. Es sollte jener Punkt gewählt werden, bei welchem ein Knick erfolgt.

## 08.03.2024

### Problematik - Sehr hochdimensionierter Datensatz

- Grundsätzlich muss ein hochdimensionierter (Dimension = Kategorie) Datensatz auf möglichst wenig Dimensionen reduziert werden, ohne dass kein essenzieller Informationsverlust entsteht
- Principal Component Analysis erlaubt Dimensionalitätsreduktion

### PCA

- Bei der PCA wird ein Vektor mit einer Matrix multipliziert
- Die Anzahl der Spalten muss genauso gross sein wie die Anzahl der Elemente im Vektor
- Die Anzahl der Zeilen kann variieren
- Am Ende stehen nicht mehr die Originalkategorien drin, es ist eine Mischung aus den einzelnen Gewichtungen der Kategorien
- Die PCA analyisiert den Datensatz und findet eine Matrix
- Daten sind in der Regel nicht gleichmässig über alle Dimensionen verteilt (2d vs 3d)
- Die PCA drückt jene Dimension zusammen (reduziert), in denen die Daten am wenigsten streuen
- Hierzu können die Punkte beispielsweise auf eine Ebene proiziert werden. Hierzu wird der orthogonale Abstand (kürzester Abstand) genutzt
- Achtung die PCA ist `kein Klassifikationsalgorithmus` sondern dient zur Vorverarbeitung der Daten

### Warum stehen die Achsen immer senkrecht zueinander?

Weil man dann nichts "kaputt" macht, d.h. wenn man diagonal zur Tür laufen würde, dann hätte man ein wenig "Weg" kaputt gemacht. Dies passiert bei orthogonalen Achsen nicht.

### Ablauf PCA

- Finde die erste Achse auf dener die Daten am meisten streuen
- Fine die zweite Achse (die muss senkrecht auf der ersten sein), auf der die Daten am zweitmeisten streuen
- Die dritte Achse muss senkdrecht auf der ersten und auf der zweiten sein

## Diskriminanzanalyse

- Erlaubt multivariate Klassifikation
- Wunsch: Die Daten sollen möglichst nahe am Mittelwert liegen, die Datencluster aber möglichst weit auseinander, sodass diese gut klassifiziert werden können.
- Alternativwerkzeug zur PCA
- Kann sowohl die Dimension reduzieren als auch klassifizieren
- Es erfordert auch nicht die Normalisierung der Daten. Eine Normalisierung ist bspw. essenziell für Machinelearning (Werte zwischen 0 und 1 normieren)
- Die Idee hinter der Diskriminanzanalyse ist, dass der Raum in dem die Daten leben in verschiedene Teilräume unterteilt werden (d.h. verschiedene Bereiche/Klassen)

## Skalarprodukt (inneres Produkt)

- Wenn der erste Vektor transponiert wird und mit dem zweiten Vektor multipliziert wird entsteht das Skalarprodukt (einzelner Wert)
- Kann genutzt werden um festzustellen ob zwei Vektoren senkrecht zueinander sind. Sind sie senkrecht, so ist das Skalarprodukt 0.
- Die Länge des Vektors, wenn dieser Vektor auf den zweiten Vektor draufproiziert werden würde.
- Vektoren **müssen zwingend gleich lang sein**
- Achtung das Ergebnis ist **nicht in Einheiten auf dem Koordinatensystem**
- Das Skalarprodukt wird bei der Diskriminanzanalyse nur dafür verwendet, dass man feststellen kann wie gross der Abstand von den einzelnen Punkten zum Ursprung (Nullpunkt) ist

## Matrix

- Matrix transponieren: Zeilen werden zu Spalten
- Symmetrisch: Eine Matrix ist dann symmetrisch, wenn nach der Transponation die gleiche Matrix rauskommt wie vor der Transponation. Dies trifft nur auf `quadratische Matrizen` zu (Stichwort Covarianz-Matrix)
- Diagonalmatrix: Es gibt nur Werte auf der Diagonalen und überall sonst Nullwerte
- Einheitsmatrix: Besteht nur aus dem Wert `1`

## Dyadisches Produkt (Äusseres Produkt)

- Wenn der zweite Vektor transponiert wird und mit dem ersten Vektor multipliziert wird entsteht das dyadische Produkt (Matrix).
- Vektoren **müssen nicht zwingend gleich lang sein**

## Tensoren

- Sind `mehrdimensionale Arrays`.
- Ist eine Art um Daten zu speichern und zu komprimieren
- Sind gut geeignet für Bild- und Videodaten
- Offenlegung latenter Information
- Dimensionalität wird über `Mode` angegeben (Vektor = 1D = 1-Mode-Tensor)
- Tensorfaktorisierung: Finde die beiden Vektoren A und B, sodass wenn ich sie multipliziere die Matrix rauskommt (Vektoren benötigen weniger Speicher da die Matrix aus den Spalten und Zeilen resultiert (Multiplikation))
- Je höher die Dimensionen werden, desto mehr Speicheraufwand wird benötigt um den Wert als Matrix zu speichern. Deshalb macht es Sinn das wieder als Vektoren abzubilden, da somit der Datenverbrauch bpsw. auf 4% reduziert werden kann
- Um aus einem Tensor eine Matrix zu bekommen braucht es die Matrix Multiplikationen (Kronecker, Kathri-Rao, Hadamar)

### Kronecker-Produkt

- Anzahl Spalten können unterschiedlich sein

### Kathri-Rao-Produkt

- Anzahl Spalten und Zeilen müssen gleich gross sein

### Hadamar-Produkt

- Anzahl Spalten und Zeilen müssen gleich gross sein

## 23.03.2024

### Mode 0 (Y-Achse - Spaltenweise)

Die Matrix wird anhand der Spalten aufgeschlüsselt:

- Spalte 1 aus 1. Schicht, Spalte 1 aus 2. Schicht, ... , Spalte 2 aus 1. Schicht

### Mode 1 (X-Achse - Zeilenweise)

- Zeilen müssen noch gedreht werden

### Mode 2 (Z-Achse)

- Elemente pro Schicht schreiben und dann als Spalten darstellen

### Tensorfaktorisierung

- Jeder Tensor kann in einen Rang-1 Tensor zerlegt werden
- `R = Rang`
- Bei der Tensorfaktorisierung muss `ausprobiert` werden
- Die Tensoren werden voneinander abgezogen, wenn die Differenz (Fehler) `0` ergibt, war die Zerlegung perfekt
- Das wird solange gemacht, bis die Differenz 0 oder einen zufriedenstellenden Wert ergibt
- Das Ergebnis der Tensorfaktorisierung sind die Tensormatrizen `A1, A2, ..., AN`

#### Tensorfaktorisierung nach Tucket

- Die Ränge können hier `pro Raumrichtung` angegeben werden
