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

* Ähnlichkeits- bzw. Distanzmetrik (definiert was "ähnlich" bedeutet). Hierzu wird oftmals der **Euklidische Abstand** (Luftlinie zwischen zwei Punkten) verwendet.
* Klassifikationsalgorithmus
* Festlegung eines Bewertungsmass (wie gut oder schlecht war die Klassifikation)
* Aufteilung der Daten in Trainings- und Testdaten
* Mehrere Durchläufe für unterschiedliche k-Werte da wir nicht wissen welcher k-Wert optimal ist
* Wählen den k-Wert welcher am besten abgeschnitten hat
* Klassifiziere die Daten anhand des gewählten k-Wertes

Beim K-Nearest-Neighbour können `Skalierungseffekte` auftreten, wenn bspw. die eine Grösse markant grösser ist als die andere, d.h. der eine Wert vom anderen dominiert wird. Um diesem Effekt entgegenzuwirken müssen die Werte auf eine Art und Weise einander angeglichen werden.

Der Algorithmus läuft folgendermassen ab:

* Wenn ein neuer Punkt hinzugefügt wird muss der Abstand zu allen anderen vorhandenen Punkten berechnet werden
* Anschliessend schaut man sich die Abstände in einem bestimmten Radius an und es gilt der Mehrheitsentscheid

### Was ist der K-Means Algorithmus?
Ist ebenfalls ein Standardalgorithmus zur Klassifikation. Der K-Means Algorithmus ist ein Algorithmus des `unsupervised Learnings` (d.h. wir kennen die Antwort nicht).

### Was besagt die Kardinalität einer Menge?
Die Kardinalität besagt, wie viele Elemente eine Menge beinhaltet.

## 02.03.2024

### Wann kann der K-Means Algorithmus angewendet werden?
Falls wir `unklassifizierte Daten` besitzen, von denen wir die Labels nicht wissen. Man möchte aber trotzdem etwas aus den Daten lernen, d.h. man möchte die Daten trotzdem in bestimmte Gruppen einteilen. Die Komplexität des Algorithmus ist `O(N)`.

### Was ist die Idee hinter dem K-Means Algorithmus?
* Klassifikation der Daten durch Clusterbildung, d.h. unter der Annahme: "Wenn die Daten nahe beieinander liegen, müssen sie eine gewisse Ähnlichkeit besitzen"
* Cluster sollen Elemente `mit ähnlichen Attributen` enthalten
* Der Anwender des Algorithmus muss angeben `wieviele Cluster` man will, d.h. `k wird vorgegeben`
* Der Algorithmus denkt sich `k beliebige Punkte im Raum aus`
* Anschliessend wird der euklidische Abstand (Pythagoras) eines jeden Punktes zu den definierten k's ermittelt
* Die Punkte werden anhand des euklidischen Abstandes den jeweiligen k's zugeordnet
* Anschliessend wird der Schwerpunkt der Cluster berechnet (zur Einfachheit wird der Mittelpunkt der Cluster berechnet)
* Anschliessend werden wieder die Abstände berechnet, bis sich der Mittelpunkt nicht mehr gross ändert (mit Änderung ist die Änderung der Klasse gemeint, nicht die Position)

### Welche Probleme können beim K-Means Algorithmus auftreten?
* Richtige Wahl des k's
* Konvergenz (Erreichung eines finalen Ergebnis) ist nicht garantiert, d.h. Algorithmus muss irgendwann abgebrochen werden (Definition maximale Anzahl von Durchläufen)
* Ob die Klassifikation der Cluster sinnvoll ist muss der Anwender entscheiden
* Der K-Means Algorithmus ist im Vergleich zu anderen Algorithmus zwar sehr einfach aber auch sehr schnell
  
### Was ist die Trägheit (Inertia)?
Ist die Summe aller Abstände von den einzelnen Punkten zum Cluster Mittelpunkt, d.h. ist die Trägheit sehr gross, sind die Punkte sehr verstreut. Es sollte jener Punkt gewählt werden, bei welchem ein Knick erfolgt.


