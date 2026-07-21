"""Language-specific guidance and demonstrations for transcript refinement."""

from dataclasses import dataclass

Example = tuple[str, str]


@dataclass(frozen=True)
class RefinementLanguageProfile:
    code: str
    name: str
    cleanup_guidance: str
    correction_guidance: str
    examples: tuple[Example, ...]


REFINEMENT_LANGUAGE_PROFILES: dict[str, RefinementLanguageProfile] = {
    "en": RefinementLanguageProfile(
        code="en",
        name="English",
        cleanup_guidance=(
            'English disfluencies can include "um", "uh", "er", "hmm", and "ah". '
            'Phrases such as "like", "you know", and "I mean" are removable only '
            "when they are empty fillers. Apply normal English capitalization and punctuation."
        ),
        correction_guidance=(
            'English correction cues can include "no wait", "actually", "scratch that", '
            '"I mean", "let me start over", and "make that".'
        ),
        examples=(
            (
                "so um yeah i was thinking like maybe we could try that new place tonight",
                "So yeah, I was thinking maybe we could try that new place tonight.",
            ),
            ("what time is it in uh tokyo right now", "What time is it in Tokyo right now?"),
            (
                "remind me to uh call mom tomorrow at three pm",
                "Remind me to call mom tomorrow at three pm.",
            ),
            (
                "write an email to um my manager saying i need to push the deadline",
                "Write an email to my manager saying I need to push the deadline.",
            ),
            (
                "the flight is at seven am no actually six am on friday",
                "The flight is at six am on Friday.",
            ),
            (
                "open package dot json then run the tests on GitHub",
                "Open package.json then run the tests on GitHub.",
            ),
            (
                "when is the API deploy in Berlin next Tuesday",
                "When is the API deploy in Berlin next Tuesday?",
            ),
            (
                "book the table for eight wait make that nine tonight",
                "Book the table for nine tonight.",
            ),
            ("tell me a joke about um databases", "Tell me a joke about databases."),
        ),
    ),
    "es": RefinementLanguageProfile(
        code="es",
        name="Spanish",
        cleanup_guidance=(
            'Spanish disfluencies can include "eh", "em", and filler uses of "este", '
            '"pues", "o sea", or "bueno". Preserve meaningful uses. Restore accents and '
            "Spanish opening question or exclamation marks when appropriate."
        ),
        correction_guidance=(
            'Spanish correction cues can include "no, espera", "mejor dicho", '
            '"en realidad", "quise decir", and "corrijo".'
        ),
        examples=(
            (
                "pues eh estaba pensando que podríamos probar ese sitio nuevo esta noche",
                "Estaba pensando que podríamos probar ese sitio nuevo esta noche.",
            ),
            ("qué hora es en eh tokio ahora", "¿Qué hora es en Tokio ahora?"),
            (
                "recuérdame eh llamar a mamá mañana a las tres",
                "Recuérdame llamar a mamá mañana a las tres.",
            ),
            (
                "escribe un correo a mi gerente diciendo que necesito mover la fecha límite",
                "Escribe un correo a mi gerente diciendo que necesito mover la fecha límite.",
            ),
            (
                "el vuelo sale a las siete no en realidad a las seis el viernes",
                "El vuelo sale a las seis el viernes.",
            ),
            (
                "abre package dot json y luego ejecuta los tests en GitHub",
                "Abre package.json y luego ejecuta los tests en GitHub.",
            ),
            (
                "cuándo es el API deploy en Berlín el próximo martes",
                "¿Cuándo es el API deploy en Berlín el próximo martes?",
            ),
            (
                "reserva la mesa para las ocho espera mejor a las nueve esta noche",
                "Reserva la mesa para las nueve esta noche.",
            ),
            ("cuéntame un chiste sobre eh bases de datos", "Cuéntame un chiste sobre bases de datos."),
        ),
    ),
    "fr": RefinementLanguageProfile(
        code="fr",
        name="French",
        cleanup_guidance=(
            'French disfluencies can include "euh", "heu", and empty filler uses of '
            '"ben", "enfin", "du coup", or "quoi". Preserve meaningful uses, accents, '
            "apostrophes, and normal French punctuation spacing."
        ),
        correction_guidance=(
            'French correction cues can include "non, attends", "en fait", "je veux dire", "plutôt", and "je corrige".'
        ),
        examples=(
            (
                "euh je pensais qu'on pourrait essayer ce nouveau restaurant ce soir",
                "Je pensais qu'on pourrait essayer ce nouveau restaurant ce soir.",
            ),
            ("quelle heure est-il euh à tokyo maintenant", "Quelle heure est-il à Tokyo maintenant ?"),
            (
                "rappelle-moi euh d'appeler maman demain à quinze heures",
                "Rappelle-moi d'appeler maman demain à quinze heures.",
            ),
            (
                "écris un mail à mon responsable pour dire que je dois repousser la date limite",
                "Écris un mail à mon responsable pour dire que je dois repousser la date limite.",
            ),
            (
                "le vol est à sept heures non en fait six heures vendredi",
                "Le vol est à six heures vendredi.",
            ),
            (
                "ouvre package dot json puis lance les tests sur GitHub",
                "Ouvre package.json puis lance les tests sur GitHub.",
            ),
            (
                "quand est le API deploy à Berlin mardi prochain",
                "Quand est le API deploy à Berlin mardi prochain ?",
            ),
            (
                "réserve la table pour huit heures non plutôt neuf heures ce soir",
                "Réserve la table pour neuf heures ce soir.",
            ),
            (
                "raconte-moi une blague sur euh les bases de données",
                "Raconte-moi une blague sur les bases de données.",
            ),
        ),
    ),
    "de": RefinementLanguageProfile(
        code="de",
        name="German",
        cleanup_guidance=(
            'German disfluencies can include "äh", "ähm", and empty filler uses of '
            '"also", "halt", or "sozusagen". Preserve meaningful particles. Apply German '
            "noun capitalization, punctuation, umlauts, and ß without rewriting compounds."
        ),
        correction_guidance=(
            'German correction cues can include "nein, warte", "eigentlich", '
            '"ich meine", "besser gesagt", and "Korrektur".'
        ),
        examples=(
            (
                "äh ich dachte wir könnten heute Abend dieses neue Restaurant ausprobieren",
                "Ich dachte, wir könnten heute Abend dieses neue Restaurant ausprobieren.",
            ),
            ("wie spät ist es äh gerade in Tokio", "Wie spät ist es gerade in Tokio?"),
            (
                "erinnere mich äh morgen um drei Mama anzurufen",
                "Erinnere mich morgen um drei, Mama anzurufen.",
            ),
            (
                "schreib meinem Manager eine E-Mail dass ich die Frist verschieben muss",
                "Schreib meinem Manager eine E-Mail, dass ich die Frist verschieben muss.",
            ),
            (
                "der Flug ist Freitag um sieben nein eigentlich um sechs",
                "Der Flug ist Freitag um sechs.",
            ),
            (
                "öffne package dot json und führe dann die tests auf GitHub aus",
                "Öffne package.json und führe dann die tests auf GitHub aus.",
            ),
            (
                "wann ist der API deploy nächsten Dienstag in Berlin",
                "Wann ist der API deploy nächsten Dienstag in Berlin?",
            ),
            (
                "reserviere den Tisch für acht nein besser für neun heute Abend",
                "Reserviere den Tisch für neun heute Abend.",
            ),
            (
                "erzähl mir einen Witz über äh Datenbanken",
                "Erzähl mir einen Witz über Datenbanken.",
            ),
        ),
    ),
    "ja": RefinementLanguageProfile(
        code="ja",
        name="Japanese",
        cleanup_guidance=(
            "Japanese disfluencies can include 「えーと」「えっと」「あの」「その」 when they "
            "serve only as hesitation. Preserve meaningful demonstratives. Use Japanese "
            "punctuation and do not impose Latin capitalization or spaces."
        ),
        correction_guidance=(
            "Japanese correction cues can include 「いや」「じゃなくて」「というか」"
            "「訂正」「違う」 when they clearly retract the previous phrase."
        ),
        examples=(
            (
                "えっと今夜あの新しい店に行ってみようと思ってる",
                "今夜、新しい店に行ってみようと思ってる。",
            ),
            ("東京はえっと今何時ですか", "東京は今何時ですか？"),
            (
                "明日の3時にえっと母に電話するようリマインドして",
                "明日の3時に母に電話するようリマインドして。",
            ),
            (
                "締め切りを延ばしたいと上司にメールを書いて",
                "締め切りを延ばしたいと上司にメールを書いて。",
            ),
            (
                "フライトは金曜日の朝7時いや6時です",
                "フライトは金曜日の朝6時です。",
            ),
            (
                "package dot jsonを開いてGitHubでtestsを実行して",
                "package.jsonを開いてGitHubでtestsを実行して。",
            ),
            (
                "来週の火曜日にベルリンでのAPI deployは何時ですか",
                "来週の火曜日にベルリンでのAPI deployは何時ですか？",
            ),
            (
                "今夜のテーブルを8時いや9時に予約して",
                "今夜のテーブルを9時に予約して。",
            ),
            ("データベースについてえっとジョークを言って", "データベースについてジョークを言って。"),
        ),
    ),
    "zh": RefinementLanguageProfile(
        code="zh",
        name="Chinese",
        cleanup_guidance=(
            "Chinese disfluencies can include “嗯”“呃”“那个” when used only as hesitation. "
            "Preserve meaningful uses. Use Chinese punctuation and do not insert Latin-style "
            "spaces or capitalization into Chinese text."
        ),
        correction_guidance=(
            "Chinese correction cues can include “不对”“不是”“应该说”“我是说” and “改成” "
            "when they clearly retract the previous phrase."
        ),
        examples=(
            ("嗯我在想今晚要不要去试试那家新店", "我在想今晚要不要去试试那家新店。"),
            ("东京那个现在几点", "东京现在几点？"),
            ("提醒我明天下午三点嗯给妈妈打电话", "提醒我明天下午三点给妈妈打电话。"),
            ("写一封邮件告诉经理我需要推迟截止日期", "写一封邮件告诉经理我需要推迟截止日期。"),
            ("航班是周五早上七点不对是六点", "航班是周五早上六点。"),
            (
                "打开package dot json然后在GitHub运行tests",
                "打开package.json，然后在GitHub运行tests。",
            ),
            ("下周二在柏林的API deploy是几点", "下周二在柏林的API deploy是几点？"),
            ("预订今晚八点不对九点的桌子", "预订今晚九点的桌子。"),
            ("讲一个关于嗯数据库的笑话", "讲一个关于数据库的笑话。"),
        ),
    ),
    "hi": RefinementLanguageProfile(
        code="hi",
        name="Hindi",
        cleanup_guidance=(
            'Hindi disfluencies can include "उम", "आ", "अं", and empty filler uses of '
            '"मतलब", "तो", or "जैसे". Preserve meaningful uses, Devanagari spelling, matras, '
            "and natural Hindi punctuation."
        ),
        correction_guidance=(
            'Hindi correction cues can include "नहीं, रुको", "असल में", "मेरा मतलब", "सुधार", and "इसके बजाय".'
        ),
        examples=(
            (
                "उम मैं सोच रहा था कि आज रात उस नई जगह को आज़माएँ",
                "मैं सोच रहा था कि आज रात उस नई जगह को आज़माएँ।",
            ),
            ("अभी उम टोक्यो में कितने बजे हैं", "अभी टोक्यो में कितने बजे हैं?"),
            (
                "मुझे कल तीन बजे उम माँ को फ़ोन करने की याद दिलाना",
                "मुझे कल तीन बजे माँ को फ़ोन करने की याद दिलाना।",
            ),
            (
                "मेरे मैनेजर को ईमेल लिखो कि मुझे समय सीमा आगे बढ़ानी है",
                "मेरे मैनेजर को ईमेल लिखो कि मुझे समय सीमा आगे बढ़ानी है।",
            ),
            (
                "फ़्लाइट शुक्रवार सुबह सात बजे है नहीं असल में छह बजे",
                "फ़्लाइट शुक्रवार सुबह छह बजे है।",
            ),
            (
                "package dot json खोलो और GitHub पर tests चलाओ",
                "package.json खोलो और GitHub पर tests चलाओ।",
            ),
            (
                "अगले मंगलवार बर्लिन में API deploy कितने बजे है",
                "अगले मंगलवार बर्लिन में API deploy कितने बजे है?",
            ),
            (
                "आज रात आठ बजे नहीं बल्कि नौ बजे की मेज़ बुक करो",
                "आज रात नौ बजे की मेज़ बुक करो।",
            ),
            ("उम डेटाबेस पर एक चुटकुला सुनाओ", "डेटाबेस पर एक चुटकुला सुनाओ।"),
        ),
    ),
}
