Absolut. Hier ist die "Gold-Master"-Version der Blaupause, inklusive des expliziten Export/Find-Buttons. Direkt zum Kopieren und Loslegen.

---

### **Product Requirements Document (PRD): Projekt "Lokal-Lehrer" (Arbeitstitel)**

#### 1. Übersicht & Leitbild

**Produkt:** Eine lokale Desktop-Anwendung (Arbeitstitel: "Lokal-Lehrer-Suite"), die den *gesamten* Workflow von der Rohdatensammlung bis zum fertig trainierten, spezialisierten und sicheren Small Language Model (SLM) abbildet.

**Leitbild:** "Ignoranz durch Design".

**Problem:** Allgemeine LLMs sind für sensible Umgebungen (z.B. Schulen) zu gefährlich (Halluzinationen, toxisches Wissen) und zu groß (Cloud-Zwang, DSGVO-Albtraum).

**Lösung:** Ein Tool, das es Fachexperten (ohne Programmierkenntnisse) ermöglicht, kleine, "dumme" Modelle (<3B Parameter) zu nehmen und ihnen *ausschließlich* das Wissen beizubringen, das sie benötigen. Wir bauen ein Werkzeug, das Sicherheit durch *Wissensbeschränkung* (CPT) und *Verhaltenssteuerung* (SFT) erreicht.

#### 2. Zielgruppe

* **Primär:** Bildungswissenschaftler, Forscher, technisch affine Lehrer, die einen Proof-of-Concept für sichere Schul-KI suchen.
* **Sekundär:** Kleine Organisationen, Kanzleien, Arztpraxen mit extremen Datenschutzanforderungen (DSGVO), die ein Modell auf eigenen Dokumenten *sicher* trainieren wollen.

#### 3. Technische Spezifikationen (Tech Stack)

| Kategorie | Wahl | Begründung |
| :--- | :--- | :--- |
| **Sprache** | Python 3.10+ | Das gesamte KI-Ökosystem (`torch`, `transformers`) basiert darauf. |
| **UI-Framework** | **Gradio** | Schnell, Python-nativ, ideal für "Daten rein, Skript raus"-Apps. Einfacher als PySide. |
| **Datenbank** | **SQLite** | Serverlos, eine einzelne Datei, perfekt für lokale Projekte. Wird unsere "Single Source of Truth". |
| **Kern-Libs** | `torch`, `transformers`, `peft` | Der De-facto-Standard für Training. |
| **Daten-Import** | `PyMuPDF`, `python-docx`, `pandas` | Zum Parsen von PDF, DOCX, CSV. |
| **Hardware-Check** | `psutil`, `torch` | Für die Heuristik (RAM, `cuda.is_available`, `mps.is_available`). |
| **Helfer-KI** | `ollama`, `openrouter` (via `litellm`) | `litellm` ist ein großartiger Wrapper, um beide API-Stile (OpenAI-kompatibel) anzusprechen. |

---

#### 4. Kernarchitektur: Die "Single Source of Truth"

Das ist das Herz der Software. Der Workflow ist linear und verlustfrei.

1.  **Tab 1 (Tresor):** Lädt Roh-Dokumente (PDFs) in eine zentrale `wissensbasis.sqlite`.
2.  **Diese `wissensbasis.sqlite` ist der Rohstoff für ALLES:**
    * Sie wird als **RAG-Datenbank** exportiert (das "Gedächtnis").
    * Sie füttert **Tab 2 (CPT)** (das "Wissensfundament").
    * Sie füttert **Tab 3 (SFT-Erstellung)** (die "Persönlichkeit").



---

#### 5. Funktionsanforderungen pro Tab

Hier ist die detaillierte Aufschlüsselung, was jeder Tab tun muss.

##### Tab 1: "Wissens-Tresor" (Daten-Ingestion & RAG-Quelle)

* **Zweck:** Rohdokumente importieren, in Text-Chunks zerlegen und in einer SQLite-DB verwalten. Das ist die Quelle für CPT, SFT und die RAG-DB.
* **UI-Elemente:**
    * `gr.File(file_count="multiple", label="Dokumente hochladen")`: Muss PDF, DOCX, TXT, CSV akzeptieren.
    * `gr.Textbox(label="Projekt/DB-Name")`: z.B. "bio\_klasse\_7".
    * `gr.Button("Dokumente verarbeiten")`.
    * `gr.Dropdown(label="Geladene Projekte")`: Zeigt alle erstellten `.sqlite`-Dateien im Arbeitsordner an.
    * `gr.Dataframe(interactive=True, label="Wissens-Chunks")`: Das Kernstück. Zeigt die DB-Inhalte an.
    * `gr.Button("Projekt-Datenbank im Ordner anzeigen")`: **(NEU)** Ein Hilfs-Button.
* **Backend-Logik:**
    1.  **Parsing:** Beim Klick auf "Verarbeiten" werden die Dokumente mit `PyMuPDF` etc. in Text umgewandelt.
    2.  **Chunking:** Der Text wird in logische Abschnitte (z.B. pro Absatz oder 512 Tokens) zerlegt.
    3.  **Speichern:** Die Chunks werden in `bio_klasse_7.sqlite` in einer Tabelle `knowledge` gespeichert (Spalten: `id`, `source_document`, `chunk_content`).
    4.  **Editieren:** Das `gr.Dataframe` MUSS `interactive=True` sein. Wenn der Nutzer einen Text-Chunk im Dataframe ändert, muss ein Event-Handler (`.change()`) getriggert werden, der ein `UPDATE`-Statement an die SQLite-DB sendet, um den extrahierten Text zu korrigieren.
    5.  **Export/Finden (NEU):** Der Button "Projekt-Datenbank im Ordner anzeigen" öffnet den System-Dateimanager (Explorer, Finder) und markiert die `bio_klasse_7.sqlite`-Datei. Das ist der "Export": Der Nutzer kann diese Datei von dort kopieren und in seine RAG-Anwendung einfügen.

##### Tab 2: "Wissens-Einbrenner" (CPT-Training)

* **Zweck:** Ein kleines Basismodell (z.B. Phi-3-mini, 1B) nehmen und ihm das Wissen aus dem "Tresor" per Continued Pre-training (CPT) einbrennen.
* **UI-Elemente:**
    * `gr.Dropdown(label="1. Wähle Wissens-Tresor")`: Lädt die `.sqlite`-DBs aus Tab 1.
    * `gr.Dropdown(label="2. Wähle Basismodell")`: Muss Ollama-Modelle listen oder einen Hugging Face-Namen (`TinyLlama/TinyLlama-1.1B`) akzeptieren.
    * `gr.Textbox(label="Neuer Modell-Name")`: z.B. "phi3-bio-7-cpt".
    * **Settings-Block:**
        * `gr.Button("Optimale Settings laden (Heuristik)")`.
        * `gr.Dropdown(label="Trainings-Modus", interactive=True)`: Wird von Heuristik gesetzt (z.B. "NVIDIA QLoRA (4-bit)" oder "Apple MPS LoRA (16-bit)").
        * Diverse `gr.Slider` / `gr.Number` für: `Epochs`, `Batch Size`, `Learning Rate`.
    * `gr.Button("CPT-TRAINING STARTEN")`.
    * `gr.Textbox(label="Trainings-Log", lines=15)`: Zeigt den `stdout`/`stderr` des Trainingsprozesses live an.
* **Backend-Logik:**
    1.  **Heuristik-Funktion:** Muss beim Klick `torch.cuda.is_available()`, `torch.backends.mps.is_available()` und `psutil.virtual_memory()` prüfen und die Settings-Dropdowns sinnvoll (sicher) voreinstellen. (Für Macs mit kleinen Modellen: 16-bit LoRA + Gradient Checkpointing).
    2.  **Datenlader:** Liest *alle* `chunk_content`-Felder aus der gewählten SQLite-DB, verkettet sie zu einem großen Text-Datensatz und bereitet ihn für das Language Modeling vor.
    3.  **Trainer:** Startet ein `transformers.Trainer`-Skript (nicht SFTTrainer!) für das CPT. Muss die LoRA/QLoRA-Settings (je nach Heuristik) anwenden.

##### Tab 3: "Persönlichkeits-Trainer" (SFT-Datenerstellung)

* **Zweck:** Mit KI-Hilfe die Trainingsdaten (`.jsonl`) für das Verhalten (SFT) erstellen, basierend auf dem Wissen aus dem "Tresor".
* **UI-Elemente (3-Spalten-Layout):**
    * **Spalte 1 (Quelle):**
        * `gr.Dropdown("Wähle Wissens-Tresor")`: Lädt die `knowledge`-Tabelle.
        * `gr.Dataframe("Wissens-Browser")`: Zeigt die Chunks. Ein Klick lädt den Text in einen Kontext-Editor.
    * **Spalte 2 (Editor):**
        * `gr.JSON(label="SFT-Dialog-Editor")`: Ein Editor, der die `{"messages": [{"role": "system", ...}]}`-Struktur vorgibt.
        * `gr.Textbox(label="SFT-Set Name")`: z.B. "bio\_lehrer\_verhalten.jsonl".
        * `gr.Button("Dialog zum Set hinzufügen & speichern")`.
    * **Spalte 3 (Assistent):**
        * `gr.Dropdown("Helfer-KI")`: Auswahl (Ollama, OpenRouter).
        * `gr.Textbox("Kontext-Chunk (aus Spalte 1)")`.
        * `gr.Button("Generiere Q&A-Fakten (aus Kontext)")`.
        * `gr.Button("Generiere Verhaltens-Dialog (z.B. 'Schüler ist verwirrt')")`.
* **Backend-Logik:**
    1.  Die "Generiere"-Buttons rufen (via `litellm`) die Helfer-KI mit spezifischen Meta-Prompts auf (z.B. "Erstelle 5 Q&A-Paare...").
    2.  Die Antwort der KI wird direkt in das `gr.JSON`-Feld (Editor) eingefügt.
    3.  "Speichern" nimmt den Inhalt des `gr.JSON`-Editors und hängt ihn an die angegebene `.jsonl`-Datei an (als JSON Line).

##### Tab 4: "Finaler Brenner" (SFT-Training)

* **Zweck:** Dem CPT-trainierten Modell (aus Tab 2) das gewünschte Verhalten (aus Tab 3) per SFT (QLoRA/LoRA) beibringen.
* **UI-Elemente:**
    * (Fast identisch zu Tab 2)
    * `gr.Dropdown(label="1. Wähle CPT-Modell")`: Lädt das Modell aus Tab 2 (z.B. "phi3-bio-7-cpt").
    * `gr.File(label="2. Lade SFT-Set (.jsonl)")`: Lädt die Datei aus Tab 3.
    * `gr.Textbox(label="Finaler Modell-Name")`: z.B. "lehrer-assistent-bio-7-final".
    * **Settings-Block:** (Gleiche Heuristik-Funktion wie in Tab 2).
    * `gr.Button("SFT-TRAINING STARTEN")`.
    * `gr.Textbox(label="Trainings-Log", lines=15)`.
* **Backend-Logik:**
    1.  Lädt das CPT-Modell als Basis.
    2.  Lädt das `.jsonl`-Datenset.
    3.  Startet ein `trl.SFTTrainer`-Skript mit den korrekten LoRA/QLoRA-Einstellungen (je nach Heuristik).
    4.  Speichert das finale (Adapter-)Modell, bereit für den Import in Ollama oder andere Inferenz-Engines.