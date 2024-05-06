from sclib.asyncio import SoundcloudAPI, Track, Playlist
import asyncio

async def mus_search(url):
    result = []
    api = SoundcloudAPI()
    playlist = await api.resolve(url)

    assert type(playlist) is Playlist

    
    for track in playlist.tracks:
        result.append({'name': track.title,
                    'artist': track.artist,
                    'album': track.label_name,
                    'playcount': track.playback_count,
                    'duration_ms': track.full_duration,
                    'explicit': track.genre,
                    'popularity': track.likes_count,
                    'key': track.tag_list,
                    'mode': track.reposts_count,
                    'time_signature': track.created_at,
                    'release_date': track.release_date})
    return result

if __name__ == '__main__':
    data = asyncio.run(mus_search('https://soundcloud.com/dasha-vorob-va/sets/lyubimye-moi-pesni-3?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing'))
    print(data)