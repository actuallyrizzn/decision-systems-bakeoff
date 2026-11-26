# Otto human confirm — Centa G-text/G-img v15 gold (2026-11-26)
**Method:** Manual read of all 68 text + 16 image captions against the 15 hub criteria. Mark: trust this over the small-model look.
## BLUF
- Agree with Venice teacher on **62/68** text (91%) and **15/16** img.
- **6 text + 1 img** labels Otto would change.
- After confirm: hubs **travel / news / gardening / games / family** are still **empty** on gold. The one teacher `family` (3027 shoe retirement) was a miss — restore **sports**.
- So this is **not** mainly "relabel until 90s appear." Most teacher labels are fine. The drawing-board piece is **coverage**: gold cannot promote a 15-way taxonomy it never exemplifies.

## Text fixes (Otto ≠ teacher)
| id | prior | teacher | Otto | why |
|---|---|---|---|---|
| `cgold-centa-3072` | business | writing-literature | **business** | Annual reports + pen = business reading habit. Teacher writing-literature is wrong; prior was right. |
| `cgold-centa-3224` | culture | writing-literature | **culture** | Sketchbook hands practice = arts/craft practice. Teacher writing-literature wrong; prior culture OK (design also defensible). |
| `cgold-centa-12107` | health | business | **science-technology** | Same Centa no-app/privacy poster as 12073. Teacher business; prior health nonsense. Align to sci-tech product stance. |
| `cgold-centa-3602` | science-technology | design | **science-technology** | Pull-up on divider + solder bench = electronics. Teacher design wrong. Img gold for same post is sci-tech — teacher inconsistent. |
| `cgold-centa-3648` | science-technology | writing-literature | **science-technology** | Lab/work notes for future self after absence. Teacher writing is caption-greedy; prior sci-tech fits author lane. Keep sci-tech. |
| `cgold-centa-3027` | sports | family | **sports** | Retiring running shoes after 800 miles. Teacher family is metaphor-bait; prior sports right. |

## Image fixes
| id | prior | teacher | Otto | why |
|---|---|---|---|---|
| `cimg-centa-img-4481` | culture | writing-literature | **culture** | Sketchbook hands — arts practice. Not writing-literature. |

## Otto label distribution (text)
- science-technology: 9
- entertainment: 8
- culture: 7
- design: 7
- sports: 7
- animals-nature: 6
- business: 6
- food: 6
- health: 6
- writing-literature: 6

## Implications
1. **Promote needle** should switch from `*_v15_teacher` to Otto-confirmed labels for the ~6 text + 1 img flips — small Acc move, not a 90s unlock.
2. **Do not** thrash lex for travel/news/gardening/games against this gold — there are no true positives to learn.
3. Next product step if 15 hubs stay: **add real gold examples** for empty hubs (or drop/merge hubs for ship until then).
4. Sci-tech magnet on wild teachers is a separate problem from gold label quality.
