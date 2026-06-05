from ptt.generic.classes import PTT
from ptt.bernoulli_bernoulli.classes import BBPTT
from ptt.ising_ising.classes import IIPTT
from ptt.potts_bernoulli.classes import PBPTT

map_sampler: dict[str, type[PTT]] = {
    "BBRBM": BBPTT,
    "PBRBM": PBPTT,
    "IIRBM": IIPTT,
    "BEBM": PTT,
    "CEBM": PTT,
    "default": PTT,
}
