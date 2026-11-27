"""Build the Russian-language report and defense deck from measured results."""
import json
import os
from pathlib import Path

from PIL import Image as Raster
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from auth_risk import NAMES, ROOT

ART = ROOT / "artifacts"
OUT = ROOT / "deliverables"
NIST = "https://csrc.nist.gov/pubs/sp/800/61/r3/final"
INK = "#162B37"
TEAL = "#1D6676"
PALE = "#E9F3F5"
GRAY = "#516771"
REPORT_TITLE = "Обнаружение подозрительных входов по журналам аутентификации"


def number(value, places=3):
    return f"{value:.{places}f}".replace(".", ",")


def percent(value):
    return number(value * 100, 1) + "%"


def count(value):
    return f"{value:,}".replace(",", " ")


def comparison(summary):
    return [
        [NAMES[name], number(results["threshold"], 2),
         *[number(results["test"][key]) for key in ["accuracy", "precision", "recall", "f1", "ap"]]]
        for name, results in summary["models"].items()
    ]


def make_report(summary):
    fonts = Path(os.environ.get("DEJAVU_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))
    for name, file in [("Text", "DejaVuSans.ttf"), ("Strong", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    doc = SimpleDocTemplate(str(OUT / "report.pdf"), pagesize=A4, leftMargin=51, rightMargin=51,
                            topMargin=55, bottomMargin=55, title=REPORT_TITLE)
    styles = {
        "title": ParagraphStyle("title", fontName="Strong", fontSize=25, leading=29,
                                textColor=colors.HexColor(INK), spaceAfter=18),
        "section": ParagraphStyle("section", fontName="Strong", fontSize=14, leading=19,
                                  textColor=colors.HexColor(TEAL), spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Text", fontSize=10.4, leading=16,
                               textColor=colors.HexColor(INK), spaceAfter=10),
        "caption": ParagraphStyle("caption", fontName="Text", fontSize=9, leading=13,
                                  textColor=colors.HexColor(GRAY), spaceAfter=11),
    }
    story = []

    def p(value, kind="body"):
        story.append(Paragraph(value, styles[kind]))

    def page(title):
        if story:
            story.append(PageBreak())
        p(title, "title")

    def table(header, rows, widths):
        cell = ParagraphStyle("cell", fontName="Text", fontSize=8.1, leading=11, textColor=colors.HexColor(INK))
        head = ParagraphStyle("head", parent=cell, fontName="Strong", textColor=colors.white)
        rows = [[Paragraph(str(value), head if i == 0 else cell) for value in row]
                for i, row in enumerate([header, *rows])]
        grid = Table(rows, colWidths=widths, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(PALE), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([grid, Spacer(1, 12)])

    def figure(name, width, caption):
        path = ART / f"{name}.png"
        with Raster.open(path) as image:
            height = width * image.height / image.width
        story.append(Image(str(path), width=width, height=height, hAlign="CENTER"))
        p(caption, "caption")

    stats = summary["models"][summary["winner"]]["test"]
    page(REPORT_TITLE)
    p("Отчёт по безопасности · Учебный проект · Тема № 5", "section")
    p("Аннотация", "section")
    p(f"Разработан офлайн-детектор, который оценивает риск отдельного события аутентификации. "
      f"Эксперимент использует {count(summary['clean'])} синтетических событий. Сравниваются модель частого класса, "
      f"логистическая регрессия и случайный лес. Логистическая регрессия выбрана по валидации. На более позднем "
      f"тестовом периоде её Precision составила {percent(stats['precision'])}, Recall — {percent(stats['recall'])}, "
      f"F1 — {number(stats['f1'])}.")
    p("1. Постановка задачи", "section")
    p("Аналитик SOC проверяет события, которые могут указывать на использование чужой учётной записи. "
      "Новое устройство, вход из другого региона и отказ MFA дают контекст для проверки. Такие же признаки могут "
      "появляться при командировке, смене телефона или обращении в поддержку. Задачу формулируем как бинарную "
      "классификацию обогащённого события с последующим ручным разбором сигнала.")
    p("Единица анализа — событие, доступное после получения результата MFA или регистрации его отсутствия. "
      "Модель принимает шесть полей и возвращает оценку, решения по порогу и факторы, влияющие на оценку. "
      "Оценка качества отвечает на три вопроса: сколько подозрительных событий система обнаруживает, какую "
      "нагрузку создают ложные сигналы и как меняются ответы при ухудшении телеметрии.")
    p("Состав системы", "section")
    p("Синтетический лог → очистка → хронологическое разбиение → обучение и выбор порога → пакетное "
      "обнаружение → оценка на позднем периоде → графики и разбор сигналов.")
    p("Исходный код, очищенный CSV, протокол метрик и команды запуска включены в репозиторий.")

    page("2. Источник и структура данных")
    p("Генератор создаёт по 240 событий в сутки за 90 дней. Время выбирается из минут соответствующего "
      "UTC-дня. Метка 0 соответствует легитимному событию, метка 1 — подозрительному сценарию по замыслу "
      "генератора. Сценарий определяет вероятности значений полей, а потому классы перекрываются.")
    table(["Поле", "Содержание", "Тип"], [
        ["failures_15m", "Число предшествовавших событию неудачных попыток за 15 минут", "Целое"],
        ["new_device", "Новое для учётной записи устройство", "Да / нет"],
        ["new_region", "Новый для учётной записи регион", "Да / нет"],
        ["off_hours", "Вход вне рабочего времени", "Да / нет"],
        ["client", "Браузер, мобильный или служебный клиент", "Категория"],
        ["mfa", "Пройдена, отклонена или недоступна", "Категория"],
        ["event_id, timestamp", "Идентификатор и время UTC", "Служебные"],
        ["scenario, label", "Сценарий генератора и истинный класс", "Разметка"],
    ], [112, 280, 100])
    p("Сценарии моделирования", "section")
    p("Легитимная часть включает обычную работу, поездки и обращения в поддержку. Подозрительная часть "
      "включает события с сериями ошибок, незнакомым контекстом и малозаметные входы. В позднем периоде "
      "генератор уменьшает долю подозрительных событий, увеличивает долю малозаметных среди них и долю "
      "поездок среди легитимных.")
    p("Данные абстрагируют уже рассчитанные признаки. В них нет последовательностей конкретных пользователей, "
      "реальных адресов, персональных данных и сетевых запросов. Поэтому эксперимент проверяет классификацию "
      "событий с готовыми признаками и чувствительность к заложенным распределениям.")

    page("3. Очистка и разведочный анализ")
    p(f"Из {count(summary['raw'])} строк после удаления повторов по event_id осталось {count(summary['clean'])}. "
      "Невалидные значения счётчика стали пропусками. Категории приведены к нижнему регистру, пробелы удалены "
      "по краям. До заполнения пропусков счётчик был неизвестен в 604 событиях, MFA — в 540.")
    table(["Выборка", "Период (UTC)", "Всего", "Класс 1", "Доля"], [
        [label, f"{split['start']} — {split['end']}", count(split["n"]), split["positive"],
         percent(split["positive"] / split["n"])]
        for label, split in zip(["Обучение", "Валидация", "Тест"], summary["splits"].values())
    ], [94, 179, 68, 76, 75])
    figure("eda", 480, "Рисунок 1. Доля класса 1 по периодам и распределение статусов MFA при обучении.")
    p("Известный результат passed чаще встречается у легитимных входов. Вместе с тем он встречается в обоих "
      "классах. Статус unavailable фиксирует наблюдаемый исход, а отдельная категория missing означает "
      "отсутствие поля в записи. Их объединение скрыло бы различие между состоянием проверки и качеством данных.")
    p("Разбиение по UTC-дням позволяет проверить модель на будущем периоде. Для обучения использованы только "
      "шесть полей, доступных на момент решения. Время применяется для разбиения. Идентификатор, сценарий и "
      "разметка исключены из входа модели.")

    page("4. Модели и протокол оценки")
    p("Преобразователи, обученные на первых 54 днях, применяются ко всем периодам. Пропуски в счётчике "
      "заполняются медианой обучающей выборки, затем столбец стандартизируется. Пропуски категорий заменяются "
      "на missing. Категории кодируются one-hot. Незнакомая категория при обработке не активирует известных "
      "категорий того же поля.")
    table(["Подход", "Роль и настройки"], [
        ["Частый класс", "Базовая модель всегда предсказывает преобладающий класс при пороге 0,50."],
        ["Логистическая регрессия", "Линейная модель, взвешенная по обратной частоте классов, до 1 000 итераций."],
        ["Случайный лес", "200 деревьев, глубина до 8, минимум 25 событий в листе, балансировка весов."],
    ], [145, 347])
    p("Для двух обучаемых моделей порог выбирается на валидации перебором от 0,05 до 0,95 с шагом 0,01. "
      "Критерий — максимум F1. При равенстве выбирается больший порог. Основная модель выбирается по F1 "
      "того же периода. Параметры предобработки, модели и пороги фиксируются до теста. Генератор и обучение "
      "используют фиксированные состояния генераторов случайных чисел.")
    p("Как читать метрики", "section")
    p("Accuracy — доля всех правильных решений. Precision — доля истинных подозрительных событий среди "
      "сигналов. Recall — доля обнаруженных подозрительных событий. F1 — гармоническое среднее Precision "
      "и Recall. Average Precision (AP) оценивает ранжирование по всей PR-кривой. Метрики Precision и Recall "
      "считаются для класса 1.")
    p("F1 подходит для учебного сравнения, поскольку учитывает качество сигналов и пропуски. Для эксплуатации "
      "порог имеет смысл выбирать по допустимой нагрузке и последствиям пропуска для конкретных учётных записей.")

    page("5. Результаты на отложенном периоде")
    table(["Модель", "Порог", "Accuracy", "Precision", "Recall", "F1", "AP"], comparison(summary),
          [139, 52, 62, 64, 59, 58, 58])
    figure("precision_recall", 440, "Рисунок 2. Precision–Recall на 4 320 событиях отложенного периода.")
    val = summary["models"]
    p(f"На валидации F1 составил {number(val['logistic']['validation']['f1'])} у логистической регрессии "
      f"и {number(val['forest']['validation']['f1'])} у случайного леса. На тесте лес обнаружил 50 событий "
      "против 44 у регрессии, но выдал 249 ложных сигналов против 178. Логистическая регрессия дала лучший F1 "
      "и AP в обоих измеренных периодах.")
    p("Высокая accuracy частого класса обусловлена дисбалансом. Он пропускает все 135 подозрительных входов. "
      "Падение F1 регрессии с 0,442 до 0,246 произошло одновременно со снижением доли положительного класса "
      "и изменением смеси сценариев. Эксперимент не выделяет индивидуальный вклад каждого из этих факторов.")

    page("6. Матрица ошибок и нагрузка")
    figure("confusion", 366, "Рисунок 3. Логистическая регрессия при пороге 0,87. Строки — истинный класс.")
    table(["Сценарий", "Событий", "Ложных сигналов", "Пропусков"],
          [[name, row["n"], row["fp"], row["fn"]] for name, row in summary["errors"].items()],
          [155, 94, 125, 118])
    p(f"Из {stats['alerts']} сигналов {stats['fp']} ложные. Поездки (travel) и поддержка (support) "
      "составляют 170 из них. Все 31 малозаметное подозрительное событие (quiet) пропущены. Также пропущено "
      "48 из 71 подозрительных входов с незнакомым контекстом.")
    p("За 18 тестовых суток детектор давал в среднем 12,3 сигнала в день, 9,9 из них ложные. Эта оценка "
      "соответствует синтетической нагрузке в 240 событий в сутки. В реальном SOC нагрузку нужно измерять "
      "на фактическом объёме и учитывать взаимосвязь событий при объединении сигналов в кейсы.")

    page("7. Объяснимость решений")
    p("Глобальную значимость оценивали на отложенной выборке. Каждый столбец независимо перемешивали "
      "восемь раз и измеряли падение AP при неизменной модели. Столбики показывают среднее падение, "
      "отрезки — стандартное отклонение между повторами.")
    figure("importance", 470, "Рисунок 4. Перестановочная важность шести входных полей.")
    p("Наибольшее среднее падение AP наблюдалось при перемешивании счётчика ошибок, признака нового устройства "
      "и нового региона. Близкие оценки и пересечение интервалов показывают, что жёсткий порядок признаков "
      "интерпретировать сложно. Важность также зависит от распределения и связей между полями.")
    p("Локальное объяснение для E20390", "section")
    p("Оценка 0,997254 превысила порог. При замене счётчика ошибок медианой обучающей выборки модель "
      "снизила оценку на 0,0972. При замене результата MFA модой снижение составило 0,0092. Значение "
      "каждого другого поля при этом фиксируется. Такие изменения помогают выбрать, какой контекст "
      "проверить первым. Они зависят от выбранного типичного значения, не складываются и не доказывают "
      "причинную связь с инцидентом.")
    p("Для ложного сигнала E20598 сильными локальными факторами стали результат MFA и регион. Аналитику "
      "понадобится сверка контекста с журналами и известной активностью пользователя.")

    page("8. Устойчивость к искажению телеметрии")
    p("Проведены два теста на изменённых копиях тестовой выборки. Изменения затрагивают только готовые признаки. "
      "Метки, модель и порог зафиксированы. Шум моделирует ошибку агрегации числа неудачных входов. Потеря "
      "контекста моделирует неполную доставку телеметрии.")
    p("В первом тесте к каждому известному счётчику добавлялось случайное целое от −2 до +2. Результаты "
      "ограничивались снизу нулём. Во втором тесте для каждого из полей new_device, new_region и mfa "
      "независимо маскировалась случайная доля ячеек с вероятностью 35%. Результаты сравниваются с исходным "
      "тестом в одинаковых условиях.")
    table(["Вход", "Precision", "Recall", "F1", "AP", "Сигналов", "Смен решений"], [
        [name, number(row["precision"]), number(row["recall"]), number(row["f1"]), number(row["ap"]),
         row["alerts"], row["flips"]] for name, row in summary["stress"].items()
    ], [140, 61, 57, 50, 50, 65, 69])
    p("Шум в счётчике увеличил число ложных сигналов со 178 до 216 и снизил AP с 0,214 до 0,197. "
      "При потере контекста Recall упал с 32,6% до 22,2%, хотя Precision выросла. Детектор обнаружил лишь "
      "30 из 135 подозрительных событий, на 14 меньше исходного теста. Сокращение потока сигналов здесь "
      "сопровождается потерей покрытия.")
    p("Проверка оценивает два контролируемых сценария искажений. При переходе к пилоту следует измерить "
      "пропуски по источникам, задержки доставки и частоту пересчёта агрегатов. Внезапный рост пропусков "
      "должен быть заметен отдельно от общего количества сигналов.")

    page("9. Применимость в SOC и вывод")
    p("Прототип подходит для проверки в теневом режиме, где модель сортирует события для аналитика. "
      "При измеренном Recall 32,6% и 178 ложных сигналах на 222 срабатывания автоматическая блокировка "
      "даст неприемлемый риск пропусков и вмешательства в легитимную работу.")
    p("Возможная интеграция", "section")
    p("Сборщик получает события IdP и MFA. Нормализация приводит поля к одной схеме, а признаки истории "
      "считаются только по событиям до момента входа. Сигнал для SIEM содержит идентификатор, время, "
      "версию модели, оценку, признаки с существенными эффектами и отметку полноты контекста. Аналитик "
      "сопоставляет сигнал с журналами, связанными входами и критичностью учётной записи.")
    p("Такая проверка и приоритизация согласуются с описанными NIST задачами триажа и реакции на инциденты. "
      "Документ NIST используется для эксплуатационного контекста, а вывод о применимости основан на "
      "результатах эксперимента. [1]")
    p("Ограничения и критерии проверки", "section")
    p("Данные происходят из одного генератора и не отражают разнообразия пользователей и инфраструктур. "
      "Метки здесь однозначны, тогда как реальные вердикты зависят от расследования. Взвешенные модельные "
      "оценки не калибровались, а качество измерялось на одном позднем периоде.")
    p("В теневом пилоте следует сравнить очередь с имеющимися правилами SOC и проверить сигналы и выборку "
      "событий ниже порога. Условия перехода к эксплуатации включают согласованный лимит сигналов в смену, "
      "целевой Recall по расследованным случаям, контроль пропусков телеметрии и проверку разных групп входов.")
    p("Источник", "section")
    p(f'[1] NIST SP 800-61 Rev. 3, Incident Response Recommendations and Considerations for Cybersecurity '
      f'Risk Management, 2025. <link href="{NIST}" color="{TEAL}">Публикация NIST</link>.', "caption")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor(PALE))
        canvas.line(51, 45, A4[0] - 51, 45)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.setFont("Text", 8)
        canvas.drawString(51, 30, "Анализ журналов аутентификации")
        canvas.drawRightString(A4[0] - 51, 30, str(doc.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def make_slides(summary):
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    deck.core_properties.title = REPORT_TITLE

    def rgb(value):
        return RGBColor.from_string(value.lstrip("#"))

    def box(slide, text, x, y, width, height, size, color=INK, bold=False):
        frame = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height)).text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        for i, line in enumerate(text.splitlines()):
            line_p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            run = line_p.add_run()
            run.text = line
            run.font.name, run.font.size = "Aptos", Pt(size)
            run.font.color.rgb, run.font.bold = rgb(color), bold
            line_p.space_after = Pt(8)

    def slide(title, section):
        page = deck.slides.add_slide(deck.slide_layouts[6])
        fill = page.background.fill
        fill.solid()
        fill.fore_color.rgb = rgb("#F4F9FA")
        box(page, section, .65, .43, 11, .34, 14, TEAL, True)
        title_size = 33 if len(title) > 47 and "\n" not in title else 37
        box(page, title, .65, 1.02, 12, 1.45, title_size, INK, True)
        return page

    def image(page, name, x, y, width):
        page.shapes.add_picture(str(ART / f"{name}.png"), Inches(x), Inches(y), width=Inches(width))

    def table(page, rows, widths, x, y, height):
        shape = page.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y),
                                      Inches(sum(widths)), Inches(height))
        for col, width in zip(shape.table.columns, widths):
            col.width = Inches(width)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                cell = shape.table.cell(i, j)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(TEAL if i == 0 else "#FFFFFF" if i % 2 else PALE)
                cell.margin_left = Inches(.13)
                for k, line in enumerate(str(value).splitlines()):
                    paragraph = cell.text_frame.paragraphs[0] if k == 0 else cell.text_frame.add_paragraph()
                    run = paragraph.add_run()
                    run.text = line
                    run.font.name, run.font.size = "Aptos", Pt(16)
                    run.font.color.rgb = rgb("#FFFFFF" if i == 0 else INK)
                    run.font.bold = i == 0

    page = slide("Обнаружение подозрительных входов\nпо журналам аутентификации", "ТЕМА № 5 · УЧЕБНЫЙ ПРОЕКТ")
    box(page, "Оценка риска события и объяснение сигнала для аналитика SOC", .7, 3.15, 10.9, 1.1, 25, TEAL)
    box(page, "21 600 синтетических событий\n90 дней журналов\n2 обучаемые модели и базовое сравнение",
        .7, 4.85, 10.9, 2.0, 22)

    page = slide("Данные имитируют работу и подозрительные входы", "ДАННЫЕ")
    image(page, "eda", .7, 2.28, 8)
    box(page, "6 полей описывают историю ошибок, устройство, регион, время, клиент и MFA.\n"
        "В позднем периоде подозрительных событий меньше.", 9.05, 2.55, 3.55, 3.8, 20)

    page = slide("Порог выбран до проверки на тесте", "ЭКСПЕРИМЕНТ")
    table(page, [
        ["Обучение", "Валидация", "Тест"], ["54 дня", "18 дней", "18 дней"],
        ["Очистка и кодирование\nОбучение двух моделей", "Выбор модели и порога\nМаксимум F1",
         "Фиксированные модели\nПоздний период со сдвигом"],
    ], [4, 4, 4], .67, 2.35, 2.7)
    box(page, "F1 на валидации  ·  Регрессия 0,442  ·  Случайный лес 0,425",
        .75, 5.7, 11.7, .7, 23, TEAL, True)

    page = slide("Регрессия дала лучший F1 на тесте", "СРАВНЕНИЕ")
    table(page, [["Модель", "Порог", "Accuracy", "Precision", "Recall", "F1", "AP"], *comparison(summary)],
          [3.2, 1.3, 1.55, 1.6, 1.45, 1.4, 1.5], .67, 2.28, 2.6)
    box(page, "Частый класс: 96,9% accuracy и 0 обнаруженных подозрительных входов.",
        .75, 5.65, 11.6, 1.1, 23)

    page = slide("178 из 222 сигналов оказались ложными", "АНАЛИЗ ОШИБОК")
    image(page, "confusion", .75, 2.1, 6.6)
    box(page, "Поездки и поддержка дали 170 ложных сигналов.\n"
        "Все 31 малозаметное подозрительное событие пропущены.", 7.7, 2.6, 4.8, 3.5, 23)

    page = slide("Модель опирается на историю ошибок и контекст", "ОБЪЯСНИМОСТЬ")
    image(page, "importance", .75, 2.45, 7.5)
    box(page, "E20390 · 0,997\nЗамена числа ошибок медианой снижает локальную оценку на 0,097.",
        8.72, 2.55, 3.8, 3.1, 22)

    page = slide("При потере контекста падает Recall", "УСТОЙЧИВОСТЬ")
    table(page, [["Изменение", "Recall", "AP", "Смен решений"], *[
        [name, percent(row["recall"]), number(row["ap"]), row["flips"]]
        for name, row in summary["stress"].items()
    ]], [5.1, 2, 2, 2.9], .67, 2.25, 2.6)
    box(page, "В тесте с потерей контекста обнаружено 30 из 135 подозрительных входов.",
        .75, 5.62, 11.6, 1.05, 22)

    page = slide("Четыре события в живом демо", "ОБНАРУЖЕНИЕ")
    box(page, "python auth_risk.py detect data/demo.csv", .75, 2.36, 11.5, .76, 24, TEAL, True)
    box(page, "E18090   Легитимный вход · нет сигнала\nE20390   Подозрительный вход · сигнал\n"
        "E20598   Ложный сигнал · чуть выше порога\nE17653   Пропуск · чуть ниже порога",
        .75, 3.45, 11.9, 2.9, 25)

    page = slide("Следующий этап — теневой пилот в SOC", "ВЫВОД")
    box(page, "Проверить качество на локальных журналах и сопоставить с действующими правилами.\n"
        "Согласовать лимит сигналов в смену и измерить пропуски.\n"
        "Передавать аналитику оценку, объяснение и полноту контекста.",
        .75, 2.45, 11.7, 3.6, 25)
    box(page, "Триаж и приоритизация · NIST SP 800-61 Rev. 3", .75, 6.62, 11.5, .45, 16, TEAL)
    deck.save(OUT / "presentation.pptx")


if __name__ == "__main__":
    results = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    make_report(results)
    make_slides(results)
