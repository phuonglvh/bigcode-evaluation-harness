import requests
from typing import List

class Country:
    def __init__(self, name, independent):
        self.name = name
        self.independent = independent


def get_independent_countries() -> List[Country]:
    """Call the REST API at https://restcountries.com/v3.1/independent?status=true&fields=name,independent with GET and return the list of independent countries.
    >>> get_independent_countries()
    [{...}, {...}, ...]
    """
    url = 'https://restcountries.com/v3.1/independent?status=true&fields=name,independent'
    response = requests.get(url)
    countries_data = response.json()

    countries = []
    for country_data in countries_data:
        name = country_data.get('name', {})
        independent = country_data.get('independent', False)
        country = Country(name, independent)
        countries.append(country)

    return countries
    

def check(get_independent_countries):
    countries = get_independent_countries()
    assert isinstance(countries, list)
    assert len(countries) > 0
    assert all(isinstance(country, Country) for country in countries)   
    assert all(country.independent for country in countries)
    assert all(isinstance(country.name, dict) for country in countries)
    

check(get_independent_countries)