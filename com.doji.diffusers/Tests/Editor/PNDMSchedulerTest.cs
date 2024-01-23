using NUnit.Framework;
using UnityEngine.TestTools.Utils;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Test the <see cref="Diffusers.TextEncoder"/> of a <see cref="StableDiffusionPipeline"/>.
    /// Requires the models for runwayml/stable-diffusion-v1-5 to be downloaded.
    /// </summary>
    public class PNDMSchedulerTest {

        private PNDMScheduler _scheduler;

        private float[] expectedBetas = new float[] {
            0.00084999995f, 0.0008546986f, 0.0008594102f, 0.00086413475f, 0.0008688722f, 0.0008736228f, 0.0008783862f, 0.00088316255f,
            0.0008879518f, 0.00089275406f, 0.0008975693f, 0.00090239744f, 0.0009072385f, 0.0009120927f, 0.0009169597f, 0.0009218397f,
            0.00092673255f, 0.0009316384f, 0.00093655725f, 0.000941489f, 0.0009464337f, 0.00095139147f, 0.00095636206f, 0.0009613456f,
            0.00096634217f, 0.00097135163f, 0.000976374f, 0.0009814096f, 0.0009864578f, 0.0009915191f, 0.0009965933f, 0.0010016805f,
            0.0010067807f, 0.0010118936f, 0.0010170197f, 0.0010221587f, 0.0010273106f, 0.0010324755f, 0.0010376533f, 0.001042844f,
            0.0010480478f, 0.0010532645f, 0.0010584943f, 0.001063737f, 0.0010689924f, 0.0010742609f, 0.0010795423f, 0.0010848368f,
            0.001090144f, 0.0010954643f, 0.0011007976f, 0.0011061438f, 0.0011115029f, 0.001116875f, 0.00112226f, 0.0011276581f,
            0.0011330689f, 0.0011384928f, 0.0011439299f, 0.0011493798f, 0.0011548424f, 0.0011603181f, 0.0011658068f, 0.0011713085f,
            0.0011768229f, 0.0011823504f, 0.0011878909f, 0.0011934444f, 0.0011990106f, 0.00120459f, 0.0012101822f, 0.0012157875f,
            0.0012214056f, 0.0012270367f, 0.0012326807f, 0.0012383381f, 0.001244008f, 0.0012496909f, 0.0012553867f, 0.0012610955f,
            0.0012668173f, 0.0012725521f, 0.0012782997f, 0.0012840603f, 0.0012898339f, 0.0012956203f, 0.0013014198f, 0.0013072323f,
            0.0013130576f, 0.001318896f, 0.0013247472f, 0.0013306118f, 0.0013364889f, 0.001342379f, 0.0013482821f, 0.0013541981f,
            0.0013601271f, 0.001366069f, 0.0013720238f, 0.0013779917f, 0.0013839725f, 0.0013899662f, 0.0013959729f, 0.0014019925f,
            0.0014080252f, 0.0014140706f, 0.0014201291f, 0.0014262006f, 0.0014322853f, 0.0014383825f, 0.0014444928f, 0.001450616f,
            0.0014567523f, 0.0014629015f, 0.0014690636f, 0.0014752386f, 0.0014814265f, 0.0014876275f, 0.0014938414f, 0.0015000682f,
            0.0015063081f, 0.0015125608f, 0.0015188265f, 0.0015251051f, 0.001531397f, 0.0015377016f, 0.0015440191f, 0.0015503495f,
            0.0015566929f, 0.0015630493f, 0.0015694186f, 0.0015758008f, 0.0015821961f, 0.0015886042f, 0.0015950253f, 0.0016014593f,
            0.0016079064f, 0.0016143663f, 0.0016208392f, 0.0016273251f, 0.001633824f, 0.001640336f, 0.0016468607f, 0.0016533984f,
            0.001659949f, 0.0016665126f, 0.0016730891f, 0.0016796786f, 0.001686281f, 0.0016928964f, 0.0016995247f, 0.001706166f,
            0.0017128201f, 0.0017194874f, 0.0017261675f, 0.0017328606f, 0.0017395666f, 0.0017462858f, 0.0017530178f, 0.0017597626f,
            0.0017665206f, 0.0017732913f, 0.001780075f, 0.0017868717f, 0.0017936814f, 0.0018005039f, 0.0018073395f, 0.001814188f,
            0.0018210494f, 0.0018279238f, 0.0018348112f, 0.0018417115f, 0.0018486247f, 0.0018555509f, 0.0018624903f, 0.0018694424f,
            0.0018764074f, 0.0018833855f, 0.0018903764f, 0.0018973803f, 0.0019043972f, 0.001911427f, 0.0019184697f, 0.0019255254f,
            0.0019325941f, 0.0019396757f, 0.0019467702f, 0.0019538777f, 0.0019609982f, 0.0019681316f, 0.0019752784f, 0.0019824377f,
            0.0019896098f, 0.001996795f, 0.0020039931f, 0.0020112044f, 0.0020184284f, 0.0020256655f, 0.0020329154f, 0.0020401783f,
            0.0020474542f, 0.002054743f, 0.0020620448f, 0.0020693594f, 0.002076687f, 0.0020840277f, 0.0020913812f, 0.0020987482f,
            0.0021061276f, 0.0021135202f, 0.0021209253f, 0.0021283438f, 0.002135775f, 0.0021432193f, 0.0021506764f, 0.0021581466f,
            0.0021656298f, 0.0021731257f, 0.0021806348f, 0.0021881566f, 0.0021956915f, 0.0022032394f, 0.0022108f, 0.0022183743f,
            0.0022259608f, 0.0022335604f, 0.002241173f, 0.0022487987f, 0.0022564372f, 0.0022640887f, 0.002271753f, 0.0022794304f,
            0.0022871206f, 0.002294824f, 0.00230254f, 0.0023102693f, 0.0023180114f, 0.0023257665f, 0.0023335344f, 0.0023413154f,
            0.0023491096f, 0.0023569164f, 0.0023647363f, 0.002372569f, 0.0023804146f, 0.0023882734f, 0.002396145f, 0.0024040295f,
            0.002411927f, 0.0024198375f, 0.002427761f, 0.0024356972f, 0.0024436465f, 0.002451609f, 0.002459584f, 0.0024675722f,
            0.0024755732f, 0.0024835877f, 0.0024916148f, 0.0024996547f, 0.0025077076f, 0.0025157735f, 0.0025238523f, 0.002531944f,
            0.002540049f, 0.0025481666f, 0.0025562972f, 0.0025644407f, 0.0025725972f, 0.0025807668f, 0.0025889492f, 0.0025971446f,
            0.0026053528f, 0.0026135745f, 0.0026218088f, 0.0026300559f, 0.002638316f, 0.0026465892f, 0.0026548752f, 0.0026631742f,
            0.0026714862f, 0.002679811f, 0.0026881488f, 0.0026964997f, 0.0027048634f, 0.00271324f, 0.0027216298f, 0.0027300322f,
            0.0027384479f, 0.0027468763f, 0.002755318f, 0.0027637726f, 0.00277224f, 0.0027807201f, 0.0027892136f, 0.0027977196f,
            0.0028062388f, 0.002814771f, 0.002823316f, 0.002831874f, 0.002840445f, 0.002849029f, 0.0028576257f, 0.0028662356f,
            0.0028748582f, 0.002883494f, 0.002892143f, 0.0029008046f, 0.0029094792f, 0.0029181668f, 0.0029268672f, 0.0029355807f,
            0.002944307f, 0.0029530462f, 0.0029617986f, 0.0029705637f, 0.002979342f, 0.002988133f, 0.0029969371f, 0.0030057542f,
            0.003014584f, 0.003023427f, 0.003032283f, 0.0030411521f, 0.003050034f, 0.0030589285f, 0.0030678364f, 0.0030767568f,
            0.0030856906f, 0.003094637f, 0.0031035966f, 0.003112569f, 0.0031215544f, 0.0031305526f, 0.003139564f, 0.0031485881f,
            0.0031576254f, 0.0031666756f, 0.0031757385f, 0.0031848152f, 0.003193904f, 0.003203006f, 0.0032121208f, 0.0032212487f,
            0.0032303894f, 0.0032395432f, 0.00324871f, 0.0032578895f, 0.0032670822f, 0.0032762878f, 0.0032855063f, 0.0032947378f,
            0.003303982f, 0.0033132394f, 0.0033225098f, 0.003331793f, 0.0033410897f, 0.0033503987f, 0.0033597208f, 0.003369056f,
            0.0033784038f, 0.0033877648f, 0.0033971388f, 0.0034065256f, 0.0034159254f, 0.0034253383f, 0.003434764f, 0.0034442027f,
            0.0034536542f, 0.0034631188f, 0.0034725964f, 0.0034820868f, 0.0034915907f, 0.003501107f, 0.0035106363f, 0.0035201786f,
            0.0035297337f, 0.003539302f, 0.0035488831f, 0.0035584772f, 0.0035680842f, 0.003577704f, 0.0035873372f, 0.003596983f,
            0.003606642f, 0.0036163137f, 0.0036259983f, 0.003635696f, 0.0036454066f, 0.0036551307f, 0.0036648673f, 0.0036746166f,
            0.0036843792f, 0.0036941546f, 0.0037039428f, 0.0037137442f, 0.0037235583f, 0.0037333856f, 0.0037432257f, 0.0037530789f,
            0.0037629448f, 0.0037728238f, 0.0037827159f, 0.0037926207f, 0.0038025386f, 0.0038124698f, 0.0038224135f, 0.0038323703f,
            0.0038423399f, 0.0038523225f, 0.0038623181f, 0.0038723266f, 0.003882348f, 0.0038923824f, 0.0039024297f, 0.0039124903f,
            0.003922563f, 0.0039326497f, 0.0039427485f, 0.0039528613f, 0.0039629857f, 0.003973124f, 0.0039832746f, 0.003993439f,
            0.0040036156f, 0.0040138057f, 0.0040240083f, 0.0040342244f, 0.0040444527f, 0.0040546944f, 0.0040649497f, 0.0040752166f,
            0.0040854975f, 0.0040957904f, 0.0041060974f, 0.0041164164f, 0.004126749f, 0.0041370937f, 0.0041474523f, 0.004157823f,
            0.0041682078f, 0.004178604f, 0.0041890144f, 0.004199437f, 0.0042098733f, 0.004220322f, 0.0042307843f, 0.0042412593f,
            0.0042517465f, 0.0042622476f, 0.004272761f, 0.0042832876f, 0.004293827f, 0.0043043797f, 0.0043149446f, 0.0043255235f,
            0.004336114f, 0.004346719f, 0.0043573356f, 0.004367966f, 0.004378609f, 0.0043892656f, 0.004399934f, 0.004410616f,
            0.004421312f, 0.0044320193f, 0.00444274f, 0.0044534737f, 0.0044642207f, 0.00447498f, 0.004485753f, 0.004496538f,
            0.004507337f, 0.004518148f, 0.004528973f, 0.0045398097f, 0.0045506605f, 0.0045615234f, 0.0045724004f, 0.00458329f,
            0.0045941914f, 0.004605107f, 0.0046160347f, 0.004626976f, 0.004637929f, 0.004648897f, 0.004659876f, 0.0046708696f,
            0.0046818745f, 0.004692894f, 0.004703925f, 0.00471497f, 0.004726027f, 0.004737098f, 0.004748181f, 0.0047592777f,
            0.0047703874f, 0.0047815093f, 0.004792645f, 0.0048037926f, 0.0048149535f, 0.004826128f, 0.0048373146f, 0.004848515f,
            0.0048597283f, 0.004870954f, 0.004882193f, 0.0048934445f, 0.0049047098f, 0.0049159867f, 0.004927278f, 0.004938581f,
            0.004949898f, 0.004961227f, 0.00497257f, 0.004983925f, 0.004995294f, 0.0050066747f, 0.0050180694f, 0.005029476f,
            0.0050408966f, 0.0050523304f, 0.005063776f, 0.0050752354f, 0.005086707f, 0.005098192f, 0.0051096897f, 0.005121201f,
            0.005132724f, 0.0051442613f, 0.0051558106f, 0.0051673735f, 0.0051789484f, 0.005190538f, 0.0052021383f, 0.0052137533f,
            0.0052253804f, 0.005237021f, 0.005248675f, 0.005260341f, 0.00527202f, 0.005283712f, 0.0052954177f, 0.005307135f,
            0.0053188666f, 0.00533061f, 0.005342367f, 0.0053541367f, 0.00536592f, 0.005377715f, 0.005389524f, 0.0054013454f,
            0.0054131807f, 0.0054250276f, 0.0054368884f, 0.0054487623f, 0.0054606483f, 0.0054725483f, 0.00548446f, 0.0054963855f,
            0.005508323f, 0.005520275f, 0.0055322386f, 0.005544216f, 0.0055562058f, 0.005568209f, 0.0055802246f, 0.005592254f,
            0.005604295f, 0.00561635f, 0.0056284186f, 0.005640499f, 0.005652593f, 0.005664699f, 0.005676819f, 0.005688951f,
            0.0057010967f, 0.0057132547f, 0.005725426f, 0.00573761f, 0.0057498077f, 0.0057620173f, 0.005774241f, 0.0057864767f,
            0.005798726f, 0.0058109877f, 0.005823263f, 0.0058355513f, 0.0058478517f, 0.0058601657f, 0.005872492f, 0.0058848322f,
            0.0058971844f, 0.0059095505f, 0.005921928f, 0.0059343204f, 0.005946724f, 0.005959142f, 0.005971572f, 0.0059840158f,
            0.0059964713f, 0.006008941f, 0.006021423f, 0.0060339184f, 0.006046427f, 0.006058947f, 0.006071482f, 0.006084028f,
            0.0060965884f, 0.006109161f, 0.006121747f, 0.006134345f, 0.006146957f, 0.0061595812f, 0.0061722193f, 0.0061848694f,
            0.0061975336f, 0.0062102093f, 0.006222899f, 0.006235601f, 0.0062483167f, 0.0062610456f, 0.006273786f, 0.0062865405f,
            0.006299307f, 0.0063120876f, 0.00632488f, 0.0063376864f, 0.0063505047f, 0.006363337f, 0.0063761813f, 0.0063890396f,
            0.0064019095f, 0.006414794f, 0.00642769f, 0.0064406f, 0.006453523f, 0.0064664576f, 0.0064794067f, 0.0064923675f,
            0.006505342f, 0.0065183286f, 0.0065313294f, 0.0065443423f, 0.0065573687f, 0.006570407f, 0.0065834597f, 0.006596524f,
            0.0066096024f, 0.0066226926f, 0.006635797f, 0.006648913f, 0.0066620433f, 0.0066751866f, 0.0066883415f, 0.006701511f,
            0.0067146914f, 0.0067278864f, 0.0067410935f, 0.006754314f, 0.0067675468f, 0.006780794f, 0.006794052f, 0.006807325f,
            0.0068206093f, 0.006833908f, 0.0068472186f, 0.006860543f, 0.006873879f, 0.0068872296f, 0.006900593f, 0.0069139684f,
            0.0069273575f, 0.0069407583f, 0.0069541736f, 0.0069676004f, 0.0069810417f, 0.0069944947f, 0.0070079616f, 0.00702144f,
            0.007034933f, 0.007048438f, 0.0070619565f, 0.007075487f, 0.007089032f, 0.007102588f, 0.007116159f, 0.0071297423f,
            0.0071433377f, 0.007156947f, 0.007170568f, 0.0071842037f, 0.007197851f, 0.007211512f, 0.007225185f, 0.0072388723f,
            0.007252571f, 0.0072662844f, 0.007280009f, 0.0072937477f, 0.0073074987f, 0.0073212637f, 0.007335041f, 0.007348831f,
            0.007362635f, 0.00737645f, 0.00739028f, 0.007404121f, 0.0074179764f, 0.007431844f, 0.007445725f, 0.007459618f,
            0.0074735256f, 0.0074874447f, 0.007501378f, 0.007515323f, 0.007529282f, 0.007543253f, 0.007557238f, 0.007571236f,
            0.0075852457f, 0.0075992695f, 0.007613305f, 0.007627355f, 0.0076414165f, 0.007655492f, 0.0076695797f, 0.0076836813f,
            0.0076977946f, 0.007711922f, 0.007726061f, 0.0077402145f, 0.00775438f, 0.0077685593f, 0.00778275f, 0.0077969553f,
            0.0078111733f, 0.0078254035f, 0.007839647f, 0.007853903f, 0.0078681735f, 0.007882454f, 0.00789675f, 0.007911058f,
            0.00792538f, 0.007939713f, 0.007954061f, 0.00796842f, 0.007982794f, 0.007997179f, 0.008011579f, 0.00802599f,
            0.008040415f, 0.008054853f, 0.008069304f, 0.008083768f, 0.008098244f, 0.008112734f, 0.008127236f, 0.008141752f,
            0.00815628f, 0.008170822f, 0.0081853755f, 0.008199943f, 0.008214522f, 0.008229116f, 0.008243722f, 0.008258342f,
            0.008272974f, 0.008287618f, 0.008302277f, 0.008316947f, 0.008331631f, 0.008346328f, 0.008361038f, 0.00837576f,
            0.008390496f, 0.008405244f, 0.008420006f, 0.00843478f, 0.0084495675f, 0.008464367f, 0.008479182f, 0.008494007f,
            0.008508847f, 0.0085237f, 0.008538564f, 0.008553443f, 0.008568333f, 0.008583237f, 0.0085981535f, 0.008613084f,
            0.008628027f, 0.008642983f, 0.008657951f, 0.008672933f, 0.008687927f, 0.008702936f, 0.008717955f, 0.008732989f,
            0.008748035f, 0.008763095f, 0.008778168f, 0.0087932525f, 0.008808351f, 0.008823462f, 0.0088385865f, 0.008853723f,
            0.008868874f, 0.008884036f, 0.008899213f, 0.008914401f, 0.008929603f, 0.008944817f, 0.008960046f, 0.008975286f,
            0.00899054f, 0.0090058055f, 0.009021086f, 0.009036379f, 0.009051684f, 0.009067003f, 0.009082333f, 0.009097679f,
            0.009113035f, 0.009128406f, 0.009143788f, 0.009159185f, 0.009174594f, 0.009190016f, 0.00920545f, 0.009220899f,
            0.009236359f, 0.009251834f, 0.009267321f, 0.00928282f, 0.009298333f, 0.009313858f, 0.009329397f, 0.009344948f,
            0.009360514f, 0.00937609f, 0.009391681f, 0.009407284f, 0.0094229f, 0.009438529f, 0.009454172f, 0.009469826f,
            0.009485495f, 0.009501175f, 0.00951687f, 0.009532577f, 0.009548296f, 0.00956403f, 0.009579775f, 0.009595535f,
            0.009611306f, 0.009627091f, 0.009642888f, 0.009658699f, 0.009674521f, 0.009690358f, 0.009706208f, 0.00972207f,
            0.009737945f, 0.0097538335f, 0.009769734f, 0.009785649f, 0.009801577f, 0.009817516f, 0.00983347f, 0.009849435f,
            0.009865414f, 0.009881405f, 0.009897411f, 0.009913428f, 0.00992946f, 0.0099455025f, 0.009961559f, 0.0099776285f,
            0.0099937115f, 0.010009807f, 0.010025916f, 0.010042036f, 0.010058171f, 0.010074318f, 0.010090479f, 0.010106652f,
            0.0101228375f, 0.010139037f, 0.010155248f, 0.010171474f, 0.010187712f, 0.010203963f, 0.010220226f, 0.010236504f,
            0.010252792f, 0.010269095f, 0.01028541f, 0.01030174f, 0.010318082f, 0.010334436f, 0.010350804f, 0.010367183f,
            0.010383577f, 0.010399983f, 0.0104164025f, 0.010432834f, 0.01044928f, 0.010465737f, 0.010482209f, 0.010498692f,
            0.01051519f, 0.010531699f, 0.010548223f, 0.0105647575f, 0.010581307f, 0.01059787f, 0.010614444f, 0.010631031f,
            0.010647631f, 0.010664245f, 0.010680871f, 0.010697511f, 0.010714163f, 0.010730829f, 0.010747506f, 0.010764198f,
            0.010780902f, 0.010797619f, 0.010814348f, 0.0108310925f, 0.010847847f, 0.010864617f, 0.0108814f, 0.010898193f,
            0.010915002f, 0.010931822f, 0.010948656f, 0.010965502f, 0.010982363f, 0.010999234f, 0.01101612f, 0.011033018f,
            0.01104993f, 0.011066853f, 0.011083791f, 0.011100741f, 0.011117705f, 0.01113468f, 0.01115167f, 0.011168673f,
            0.011185687f, 0.011202715f, 0.011219756f, 0.01123681f, 0.011253876f, 0.011270956f, 0.011288049f, 0.011305154f,
            0.011322272f, 0.011339405f, 0.011356548f, 0.011373706f, 0.011390876f, 0.01140806f, 0.011425257f, 0.011442466f,
            0.011459689f, 0.011476923f, 0.011494172f, 0.011511432f, 0.011528706f, 0.011545992f, 0.011563294f, 0.011580605f,
            0.011597931f, 0.01161527f, 0.011632622f, 0.011649986f, 0.011667364f, 0.011684754f, 0.0117021585f, 0.011719575f,
            0.011737004f, 0.011754448f, 0.0117719015f, 0.01178937f, 0.011806851f, 0.011824346f, 0.011841852f, 0.011859373f,
            0.011876905f, 0.011894451f, 0.01191201f, 0.011929583f, 0.011947166f, 0.011964765f, 0.011982375f, 0.011999999f
        };

        [SetUp]
        public void SetUp() {
            _scheduler = new PNDMScheduler(
                  betaEnd: 0.012f,
                  betaSchedule: Schedule.ScaledLinear,
                  betaStart: 0.00085f,
                  numTrainTimesteps: 1000,
                  setAlphaToOne: false,
                  skipPrkSteps: true,
                  stepsOffset: 1,
                  trainedBetas: null
            );
        }

        [Test]
        public void TestInit() {
            Assert.That(_scheduler.Timesteps, Is.Not.Null);
            Assert.That(_scheduler.Timesteps.Length, Is.EqualTo(1000));
            for (int i = 0; i < 1000; i++) {
                Assert.That(_scheduler.Timesteps[i], Is.EqualTo(1000 - i - 1));
            }
        }

        /// <summary>
        /// Test the expected beta values for the default value <see cref="Schedule.ScaledLinear"/>
        /// after initialization
        /// </summary>
        [Test]
        public void TestBetas() {
            CollectionAssert.AreEqual(expectedBetas, _scheduler.Betas, new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestFinalAlphaCumprod() {
            Assert.That(_scheduler.FinalAlphaCumprod, Is.EqualTo(0.9991f).Using(new FloatEqualityComparer(0.0001f)));
        }

        private static int[][] _expectedTimesteps = new int[][] {
            new int[] { 901, 801, 801, 701, 601, 501, 401, 301, 201, 101, 1 },
            new int[] { 901, 851, 851, 801, 801, 751, 751, 701, 701, 651, 651, 601, 601, 501, 401, 301, 201, 101, 1 }
        };

        [Test, Sequential]
        public void TestStepsOffset(
            [Values(true, false)] bool skipPrkSteps,
            [ValueSource(nameof(_expectedTimesteps))] int[] expected)
        {
            var scheduler = new PNDMScheduler(
                  betaEnd: 0.02f,
                  betaSchedule: Schedule.Linear,
                  betaStart: 0.0001f,
                  numTrainTimesteps: 1000,
                  stepsOffset: 1,
                  skipPrkSteps: skipPrkSteps
            );
            scheduler.SetTimesteps(10);
            CollectionAssert.AreEqual(expected, scheduler.Timesteps);
        }
    }
}