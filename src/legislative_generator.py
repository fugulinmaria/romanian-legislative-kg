"""
Romanian Legislative Text Generator
Generates synthetic Romanian legislative texts for knowledge graph construction.
"""

import random


class RomanianLegislativeGenerator:
    """Generates realistic Romanian legislative documents."""

    def __init__(self):
        """Initialize the legislative generator with templates and data."""
        self.law_types = ["Lege", "Lege organică", "Ordonanță de urgență", "Ordonanță"]
        self.ministries = [
            "Ministerul Cercetării, Inovării și Digitalizării",
            "Ministerul Educației",
            "Ministerul Sănătății",
            "Ministerul Finanțelor Publice",
            "Ministerul Justiției",
            "Ministerul Transporturilor",
            "Ministerul Mediului",
        ]
        self.presidents = ["Klaus Iohannis", "Președintele României"]
        self.locations = ["București", "Palatul Parlamentului", "Palatul Cotroceni"]
        self.publishers = ["Monitorul Oficial al României"]

        self.relations = [
            "emis_de",
            "modifică",
            "promulgat_de",
            "publicat_în",
            "are_sediul_în",
            "responsabil_pentru",
            "abroga",
            "completează",
        ]

        self.topics = [
            "digitalizarea administrativă",
            "protecția mediului",
            "educația națională",
            "sănătatea publică",
            "transportul rutier",
            "administrația fiscală",
            "justiția",
            "cercetarea științifică",
        ]

    def generate_law_number(self, year=None):
        """Generate a realistic law number."""
        if year is None:
            year = random.randint(2020, 2026)
        number = random.randint(1, 500)
        return f"nr. {number}/{year}"

    def generate_simple_law(self, topic=None, num_articles=6):
        """
        Generate a simple legislative text.

        Args:
            topic (str, optional): Law topic
            num_articles (int): Number of articles to generate

        Returns:
            str: Generated legislative text
        """
        if topic is None:
            topic = random.choice(self.topics)

        law_type = random.choice(self.law_types)
        law_number = self.generate_law_number()
        ministry = random.choice(self.ministries)
        president = random.choice(self.presidents)
        location = random.choice(self.locations)
        publisher = random.choice(self.publishers)

        text = f"""{law_type} {law_number} privind {topic}.

Articolul 1: Prezenta lege este emisă de Parlamentul României.
Articolul 2: {ministry} este responsabil pentru implementarea normelor.
Articolul 3: Prezenta lege modifică Ordonanța de Urgență nr. {random.randint(10, 100)}/{random.randint(2015, 2023)}.
Articolul 4: {president}, în calitate de Președinte, a promulgat această lege în {location}.
Articolul 5: Legea a fost publicată în {publisher}.
Articolul 6: Parlamentul României are sediul în Palatul Parlamentului.
"""
        return text

    def generate_amendment_law(self, target_law_number=None):
        """
        Generate a law that modifies another law.

        Args:
            target_law_number (str, optional): Law to be modified

        Returns:
            str: Generated amendment legislative text
        """
        if target_law_number is None:
            target_law_number = f"nr. {random.randint(1, 300)}/{random.randint(2015, 2023)}"

        law_number = self.generate_law_number()
        ministry = random.choice(self.ministries)
        president = random.choice(self.presidents)

        text = f"""Lege {law_number} pentru modificarea și completarea Legii {target_law_number}.

Articolul 1: Prezenta lege este emisă de Parlamentul României.
Articolul 2: Legea {law_number} modifică Legea {target_law_number}.
Articolul 3: La articolul 5 din Legea {target_law_number}, alineatul (2) se abrogă.
Articolul 4: Legea {law_number} completează Legea {target_law_number} cu noi prevederi.
Articolul 5: {ministry} este responsabil pentru aplicarea prezentei legi.
Articolul 6: {president} a promulgat această lege în București.
Articolul 7: Prezenta lege a fost publicată în Monitorul Oficial al României.
"""
        return text

    def generate_emergency_ordinance(self, topic=None):
        """
        Generate an emergency ordinance (Ordonanță de urgență).

        Args:
            topic (str, optional): Ordinance topic

        Returns:
            str: Generated emergency ordinance text
        """
        if topic is None:
            topic = random.choice(self.topics)

        ordinance_number = f"nr. {random.randint(1, 150)}/{random.randint(2020, 2026)}"
        ministry = random.choice(self.ministries)

        text = f"""Ordonanță de urgență {ordinance_number} privind {topic}.

Articolul 1: Prezenta ordonanță este emisă de Guvernul României.
Articolul 2: {ministry} coordonează implementarea măsurilor prevăzute.
Articolul 3: Ordonanța de urgență {ordinance_number} modifică Legea nr. {random.randint(50, 250)}/{random.randint(2010, 2020)}.
Articolul 4: Prezenta ordonanță intră în vigoare la data publicării în Monitorul Oficial al României.
Articolul 5: Guvernul României are sediul în București.
"""
        return text

    def generate_complex_law(self, topic=None, num_articles=12):
        """
        Generate a more complex law with chapters and multiple relations.

        Args:
            topic (str, optional): Law topic
            num_articles (int): Number of articles

        Returns:
            str: Generated complex legislative text
        """
        if topic is None:
            topic = random.choice(self.topics)

        law_number = self.generate_law_number()
        ministry1 = random.choice(self.ministries)
        ministry2 = random.choice([m for m in self.ministries if m != ministry1])
        president = random.choice(self.presidents)

        old_law1 = f"nr. {random.randint(50, 200)}/{random.randint(2010, 2020)}"
        old_law2 = f"nr. {random.randint(50, 200)}/{random.randint(2010, 2020)}"

        text = f"""Lege {law_number} privind {topic}.

CAPITOLUL I - Dispoziții generale

Articolul 1: Prezenta lege este emisă de Parlamentul României.
Articolul 2: Scopul prezentei legi îl constituie reglementarea {topic}.
Articolul 3: {ministry1} este responsabil pentru coordonarea aplicării prezentei legi.
Articolul 4: {ministry2} colaborează cu {ministry1} în implementarea normelor.

CAPITOLUL II - Modificări legislative

Articolul 5: Prezenta lege modifică Legea {old_law1}.
Articolul 6: Prezenta lege modifică Legea {old_law2}.
Articolul 7: La articolul 10 din Legea {old_law1}, alineatul (3) se abrogă.
Articolul 8: Legea {old_law2} se completează cu un nou articol.

CAPITOLUL III - Dispoziții finale

Articolul 9: {president} a promulgat această lege în București.
Articolul 10: Prezenta lege a fost publicată în Monitorul Oficial al României.
Articolul 11: Parlamentul României are sediul în Palatul Parlamentului.
Articolul 12: Prezenta lege intră în vigoare la 30 de zile de la publicare.
"""
        return text

    def generate_batch(self, count=5, law_types=None):
        """
        Generate multiple legislative texts.

        Args:
            count (int): Number of laws to generate
            law_types (list, optional): Types of laws to generate

        Returns:
            list: List of tuples (law_id, law_text)
        """
        if law_types is None:
            law_types = ["simple", "amendment", "emergency", "complex"]

        laws = []
        for i in range(count):
            law_type = random.choice(law_types)

            if law_type == "simple":
                text = self.generate_simple_law()
                law_id = f"simple_law_{i + 1}"
            elif law_type == "amendment":
                text = self.generate_amendment_law()
                law_id = f"amendment_law_{i + 1}"
            elif law_type == "emergency":
                text = self.generate_emergency_ordinance()
                law_id = f"emergency_ordinance_{i + 1}"
            else:  # complex
                text = self.generate_complex_law()
                law_id = f"complex_law_{i + 1}"

            laws.append((law_id, text))

        return laws

    def get_predefined_sample(self):
        """Get a predefined high-quality sample for testing."""
        return """Legea nr. 450/2024 privind digitalizarea administrativă.

CAPITOLUL I - Dispoziții generale

Articolul 1: Prezenta lege este emisă de Parlamentul României.
Articolul 2: Ministerul Cercetării, Inovării și Digitalizării este responsabil pentru implementarea normelor.
Articolul 3: Ministerul Educației colaborează cu Ministerul Cercetării, Inovării și Digitalizării.

CAPITOLUL II - Modificări și completări

Articolul 4: Prezenta lege modifică Ordonanța de Urgență nr. 57/2019.
Articolul 5: Legea nr. 450/2024 completează Legea nr. 100/2018 privind administrația publică digitală.
Articolul 6: La articolul 15 din Ordonanța de Urgență nr. 57/2019, alineatul (2) se abrogă.

CAPITOLUL III - Dispoziții finale

Articolul 7: Klaus Iohannis, în calitate de Președinte, a promulgat această lege în București.
Articolul 8: Legea a fost publicată în Monitorul Oficial al României.
Articolul 9: Parlamentul României are sediul în Palatul Parlamentului.
Articolul 10: Ministerul Cercetării, Inovării și Digitalizării are sediul în București.
Articolul 11: Prezenta lege intră în vigoare la 30 de zile de la publicare.
"""
