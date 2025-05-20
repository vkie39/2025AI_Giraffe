# 런타임 안끊기도록 1분마다 웹페이지의 리소스 확인 버튼을 눌러주는 코드
# f12 -> console의 맨 아래 > 옆에 직접 적은 후 실행 (복붙 안되더라)
'''
function tryClickConnect(){
    try {
        const button = document.querySelector("colab-connect-button");
        if(button && button.shadowRoot){
            const connectBtn = button.shadowRoot.querySelector("#connect");
            if(connectBtn){
                connectBtn.click();
                console.log("Connect Clicked");
            } else {
                console.log("connect button not found");
            }
        } else {
            console.log("colab-connect-button or shadowRoot not ready");
        }
    } catch(e) {
        console.log("Error:", e);
    }
}

setInterval(tryClickConnect, 60000);
'''