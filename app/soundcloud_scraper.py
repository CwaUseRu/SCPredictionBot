from sclib.asyncio import SoundcloudAPI, Track, Playlist
import asyncio

async def mus_search(url):
    result = []
    api = SoundcloudAPI()
    playlist = await api.resolve(url)

    assert type(playlist) is Playlist

    
    for track in playlist.tracks:
        result.append({
                        'artist': track.artist,
                        'duration': track.full_duration,
                        'genre': track.genre,
                        'label': track.label_name,
                        'tags': track.tag_list,
                        'title': track.title,
                        'album': track.album,
                        'date': track.created_at,
                        'likes': track.likes_count,
                        'stream': track.playback_count
                    })
    return result

if __name__ == '__main__':
    data = asyncio.run(mus_search('https://soundcloud.com/dasha-vorob-va/sets/lyubimye-moi-pesni-3?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing'))
    print(data)