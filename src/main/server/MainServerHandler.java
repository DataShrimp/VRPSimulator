import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.*;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.util.AsciiString;
import io.netty.util.CharsetUtil;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;

import static io.netty.handler.codec.http.HttpResponseStatus.OK;
import static io.netty.handler.codec.http.HttpVersion.HTTP_1_1;

public class MainServerHandler extends ChannelInboundHandlerAdapter{
    private static final AsciiString CONTENT_TYPE = new AsciiString("Content-Type");
    private static final AsciiString CONTENT_LENGTH = new AsciiString("Content-Length");
    private static final AsciiString CONNECTION = new AsciiString("Connection");
    private static final AsciiString KEEP_ALIVE = new AsciiString("keep-alive");

    // 初始化逻辑
    private Experiment exp = null;
    private SimuEngine engine = null;
    public MainServerHandler() {
        engine = new SimuEngine();
        exp = new Experiment();
    }

    // 主要的服务端处理逻辑
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {

        if (msg instanceof FullHttpRequest) {
            FullHttpRequest req = (FullHttpRequest)msg;
            JSONObject requestJson = null;
            JSONObject responseJson = new JSONObject();
            int n = 0;
            ArrayList<Integer> action = new ArrayList<>();

            try {
                String reqStr = parseJosnRequest(req);
                //System.out.println(req);
                if (!reqStr.isEmpty()) {
                    requestJson = new JSONObject(reqStr);
                    System.out.println(requestJson.toString());
                    n = (int)requestJson.get("n");
                    JSONArray jArray = (JSONArray) requestJson.get("action");
                    for (int i=0; i<jArray.length(); i++) {
                        action.add(jArray.getInt(i));
                    }
                }
            } catch(Exception e) {
                ResponseJson(ctx, req, new String("error: "+e.toString()));
                return;
            }

            if (n==0)
                return;

            // uri路由
            if (req.uri().equals("/start")) {
                exp.setSize(n);
                engine.initialize(exp);
                exp.setAction(action);
                engine.run();
                responseJson.put("City", exp.getCities());
                responseJson.put("Reward", exp.getReward());
            }

            if (req.uri().equals("/run")) {
                ;
            }

            ResponseJson(ctx, req, responseJson.toString());
        }

    }

    @Override
    public void channelReadComplete(ChannelHandlerContext ctx) {
        ctx.flush();
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }

    private String parseJosnRequest(FullHttpRequest request) {
        ByteBuf jsonBuf = request.content();
        String jsonStr = jsonBuf.toString(CharsetUtil.UTF_8);
        return jsonStr;
    }

    private void ResponseJson(ChannelHandlerContext ctx, FullHttpRequest req ,String jsonStr)
    {

        boolean keepAlive = HttpUtil.isKeepAlive(req);
        byte[] jsonByteByte = jsonStr.getBytes();
        FullHttpResponse response = new DefaultFullHttpResponse(HTTP_1_1, OK, Unpooled.wrappedBuffer(jsonByteByte));
        response.headers().set(CONTENT_TYPE, "text/json");
        response.headers().setInt(CONTENT_LENGTH, response.content().readableBytes());

        if (!keepAlive) {
            ctx.write(response).addListener(ChannelFutureListener.CLOSE);
        } else {
            response.headers().set(CONNECTION, KEEP_ALIVE);
            ctx.write(response);
        }
    }
}
