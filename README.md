# Mulberry Bush (2026)

**Watch it: https://toontalk.github.io/mulberry-bush-2026/**  
**How it was made: https://toontalk.github.io/mulberry-bush-2026/making-of.html**  
**Reviews of the 1976 film and its nine AI remakes: https://toontalk.github.io/mulberry-bush-2026/reviews.html**

A round for four dancers, after the 1976 film and dance *Mulberry Bush* by Ken Kahn.

Seen from above, four coloured lanterns walk in from the corners and circle a bush that grows towards their light. The bush leafs, fruits and is stripped by a whirlwind. The dancers fall down, the night passes, and they leave by the corners they came in from as the sun rises.

Nothing in `index.html` is a recording.

- **Picture:** every frame is drawn as you watch, with WebGL2, from the grown bush, the 1,392 footprints and the choreography of the rendered film.
- **Sound:** the four-voice round, drone, footsteps, berries, crickets and birds are synthesised in Web Workers when the page opens, from the film's own score of 3,413 sounds.

It is a single self-contained file of about 380 KB, and needs a browser with WebGL 2.

## The idea: turn the 1976 piece inside out

| 1976 | 2026 |
|---|---|
| Side view of a backlit screen | Looking straight down from above the bush |
| Light behind; dancers are **shadows** | The dancers **are the light**: four coloured lanterns |
| The dancers are drawn (silhouette, stripes, dots, discs) | The dancers are never drawn. You see only their light and their **bare footprints** |
| The bush is the name of the dance | The bush **grows because they dance**. Its shoots bend toward the circling lanterns, so it grows in a spiral |
| Figures cast shadows on the screen | The bush casts **four coloured shadows** that turn as the dancers go round |
| Music in C, continuous | A **four-voice round** in 6/8, one voice per lantern, each panned with its dancer |

No frame of the original was traced or reused.

## Source

Everything that makes the film is in [`source/`](source/).

| File | What it does |
|---|---|
| `timeline.py` | The single musical clock (bars of 6/8, with the accelerando) and all the choreography |
| `world.py` | Grows the bush toward the lanterns and places leaves, berries, stains, footprints and dew glints |
| `render.py` | Draws every frame: ground, four point lights and their shadows, bush, trails, lettering |
| `audio.py` | Synthesises the round, drone, footsteps, berries, wind, crickets and birds |
| `web_export.py` | Packs the world and records the score for the web page |
| `web/` | The web version: clock, WebGL renderer, ground and sound workers, player, page template |
| `build_web.py` | Assembles the single-file page |

### Building the web page

```
cd source
pip install -r requirements.txt
python world.py        # grows the bush                    -> build/world.pkl
python web_export.py   # world + score                     -> build/web/data.b64
python build_web.py    # -> "Mulberry Bush 2026 (live).html"
```

Open the HTML file in a browser. It is the same page as `index.html` here. `world.py` is seeded, so it grows the same bush every time.

### Rendering the film as an MP4

```
python render.py test 40 90 150   # stills at those seconds  -> build/test/
python render.py all              # all 5,394 frames on 8 processes -> build/video.mp4
python audio.py                   # the soundtrack           -> build/audio.wav
```

Then join `build/video.mp4` and `build/audio.wav` with ffmpeg. Two things are hard-coded at the top of `render.py`:

- **ffmpeg** is expected at `source/tools/ffmpeg.exe` (`FF`). It is only needed for `render.py all`.
- **The lettering** uses Windows' Garamond (`GARA.TTF` and `GARAIT.TTF` in `FONT_DIR`). On another system, point these at a Garamond you have.

The full render took about 40 minutes on an 8-core machine.

## Licence

The code is released under the [MIT licence](LICENSE).

Images and music written in code by Claude, 2026.
